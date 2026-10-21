"""Small diagonal GMMs, grouped validation selection and reproducible audits."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .data import balanced_groups, digest, sample_manifest, write_json


def log_density(values, model):
    weights, means, variance = (np.asarray(model[key], dtype=float) for key in ("weights", "means_ab", "variances_ab"))
    delta = values[:, None, :] - means[None]
    log_components = (np.log(weights)[None] - .5*(2*math.log(2*math.pi)
                      + np.log(variance).sum(1)[None] + (delta*delta/variance[None]).sum(2)))
    return logsumexp(log_components, axis=1)


def fit_mixture(values, components, variance_floor=4., seed=42, max_iter=100):
    """Bounded EM; initialization and fitting see training samples only."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < components*8 or not np.isfinite(x).all():
        raise ValueError("Insufficient/invalid training chroma samples")
    if not 1 <= components <= 8 or not np.isfinite(variance_floor) or variance_floor <= 0 or max_iter < 1:
        raise ValueError("Invalid mixture capacity/regularization")
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(len(x))]]
    for _ in range(1, components):
        d2 = ((x[:, None]-np.asarray(means)[None])**2).sum(2).min(1)
        index = rng.choice(len(x), p=d2/d2.sum()) if d2.sum() > 0 else rng.integers(len(x))
        means.append(x[index])
    means = np.asarray(means).copy()
    variance = np.tile(np.maximum(x.var(0), variance_floor), (components, 1))
    weights = np.full(components, 1/components)
    previous, history = -float("inf"), []
    for _ in range(max_iter):
        logp = np.log(weights)[None] - .5*(2*math.log(2*math.pi)+np.log(variance).sum(1)[None]
                    + ((x[:, None]-means[None])**2/variance[None]).sum(2))
        total = logsumexp(logp, axis=1)
        likelihood = float(total.mean()); history.append(likelihood)
        if len(history) > 1 and abs(likelihood-previous) < 1e-5:
            break
        previous = likelihood
        responsibility = np.exp(logp-total[:, None])
        count = responsibility.sum(0).clip(1e-8)
        weights = np.maximum(count/count.sum(), 1e-8); weights /= weights.sum()
        means = responsibility.T @ x / count[:, None]
        variance = (responsibility[:, :, None]*(x[:, None]-means[None])**2).sum(0)/count[:, None]
        variance = np.maximum(variance, variance_floor)
    order = np.lexsort((means[:, 1], means[:, 0]))
    return dict(weights=weights[order].tolist(), means_ab=means[order].tolist(),
                variances_ab=variance[order].tolist(), iterations=len(history),
                train_nll=float(-log_density(x,dict(weights=weights, means_ab=means, variances_ab=variance)).mean()))


def group_scores(groups, model):
    values = {group:float(-log_density(x,model).mean()) for group,x in groups.items()}
    a = np.asarray(list(values.values()))
    return dict(mean_group_nll=float(a.mean()), standard_error=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.,
                groups=values)


def fit_bank(manifest_path, output, config):
    if Path(output).exists():
        raise ValueError("Preserve existing learned artifact")
    required = ("components", "variance_floor", "min_train_groups", "min_val_groups", "min_group_pixels",
                "image_cap", "group_cap", "seed", "bootstrap_runs")
    if any(key not in config for key in required):
        raise ValueError("Use an explicit fit configuration")
    config = dict(config)
    config.setdefault("support_quantile", .99)
    if not 0 < config["support_quantile"] < 1:
        raise ValueError("Color support quantile must lie strictly between zero and one")
    if config["min_train_groups"] < 2 or config["min_val_groups"] < 2 or not 0 <= config["bootstrap_runs"] <= 20:
        raise ValueError("Grouped fitting/selection needs at least two groups on each side")
    components = sorted(set(config["components"]))
    if not components or any(not isinstance(k,int) or not 1<=k<=8 for k in components):
        raise ValueError("Provide a small predeclared component candidate set")
    manifest, pools, identities = sample_manifest(manifest_path,config["image_cap"],config["group_cap"],config["seed"])
    classes = {}
    for index,name in enumerate(manifest["class_names"]):
        train_groups, x = balanced_groups(pools,"train",name,config["min_group_pixels"])
        val_groups, _ = balanced_groups(pools,"val",name,config["min_group_pixels"])
        support = dict(train_groups=len(train_groups),val_groups=len(val_groups),train_pixels=len(x),
                       val_pixels=sum(len(v) for v in val_groups.values()))
        if len(train_groups)<config["min_train_groups"] or len(val_groups)<config["min_val_groups"]:
            classes[name] = dict(status="unsupported",support=support,reason="insufficient independent groups; no hand-color fallback")
            continue
        candidates = []
        for k in components:
            if len(x) < k*32:
                continue
            models = [fit_mixture(x,k,config["variance_floor"],config["seed"]+index*100+j) for j in range(2)]
            model = min(models,key=lambda m:m["train_nll"])
            score = group_scores(val_groups,model)
            candidates.append(dict(components=k,model=model,validation=score))
        if not candidates:
            classes[name] = dict(status="unsupported",support=support,reason="insufficient pixels for declared candidates")
            continue
        best = min(candidates,key=lambda c:c["validation"]["mean_group_nll"])
        cutoff = best["validation"]["mean_group_nll"] + best["validation"]["standard_error"]
        chosen = min((c for c in candidates if c["validation"]["mean_group_nll"] <= cutoff),key=lambda c:c["components"])
        model = chosen["model"]
        # Define the acceptable density using balanced TRAIN pixels only. Validation
        # chooses K, but never supplies color targets or this loss threshold.
        train_nll = -log_density(x, model)
        support_nll = float(np.quantile(train_nll, config["support_quantile"]))
        loss_support = dict(source="group_balanced_train_only", quantile=config["support_quantile"],
                            max_nll=support_nll, train_fraction=float((train_nll <= support_nll).mean()),
                            validation_fraction=float(np.mean(np.concatenate([
                                -log_density(values, model) <= support_nll for values in val_groups.values()]))))
        rng = np.random.default_rng(config["seed"]+index)
        bootstrap = []
        groups = list(train_groups)
        for trial in range(config["bootstrap_runs"]):
            sample = np.concatenate([train_groups[groups[i]] for i in rng.integers(len(groups),size=len(groups))])
            replica = fit_mixture(sample,chosen["components"],config["variance_floor"],config["seed"]+trial)
            bootstrap.append(group_scores(val_groups,replica)["mean_group_nll"])
        classes[name] = dict(status="supported", **model, support=support, selected_components=chosen["components"],
                             loss_support=loss_support,
                             validation=chosen["validation"], train_validation_gap=chosen["validation"]["mean_group_nll"]-model["train_nll"],
                             candidate_validation=[dict(components=c["components"],**c["validation"]) for c in candidates],
                             bootstrap_validation_nll=bootstrap)
    if not any(c["status"]=="supported" for c in classes.values()):
        raise ValueError("No class has enough training/validation support; add data, do not invent colors")
    bank = dict(format="learned_chroma_gmm_v1", class_names=manifest["class_names"], classes=classes,
                provenance=dict(dataset_id=manifest.get("dataset_id","unspecified"), manifest_sha256=digest(manifest_path),
                                samples=identities, teacher=manifest.get("teacher",{})),
                fit_config=config, selection="smallest K within one standard error of best group-mean validation NLL",
                limits=["Validation is used for model selection, not an independent test.",
                        "Only RGB train/val roles read; no test pixels used in fitting.",
                        "Pseudo-label errors and dataset bias remain; cross-dataset gains unverified."])
    write_json(output,bank)
    return bank


def evaluate_bank(artifact, manifest_path, output, role="test"):
    if role not in ("val","test"):
        raise ValueError("Evaluate on val/test roles")
    bank = json.loads(Path(artifact).read_text())
    cfg=bank['fit_config']
    manifest,pools,identities = sample_manifest(manifest_path,cfg['image_cap'],cfg['group_cap'],cfg['seed'],roles=(role,))
    if manifest["class_names"] != bank["class_names"]:
        raise ValueError("Align canonical class names before evaluation")
    old = bank["provenance"]
    forbidden = [r for r in old["samples"] if r["split"]=="train" or role=="test"]
    seen_pixels={r["rgb_content_sha256"] for r in forbidden}
    seen_groups={str(r["group"]) for r in forbidden} if manifest.get("dataset_id")==old["dataset_id"] else set()
    if any(r["rgb_content_sha256"] in seen_pixels or str(r["group"]) in seen_groups for r in identities):
        raise ValueError("Evaluation overlaps fitted/selection data")
    scores={}
    for name in manifest["class_names"]:
        groups,_=balanced_groups(pools,role,name,bank["fit_config"]["min_group_pixels"])
        if bank["classes"][name]["status"]=="supported" and groups:
            scores[name]=group_scores(groups,bank["classes"][name])
    result=dict(prior_sha256=digest(artifact),dataset_id=manifest.get("dataset_id"),role=role,scores=scores,
                limits="RGB prior density only; no TIR colorization or generalization proof. val may have selected K.")
    write_json(output,result)
    return result
