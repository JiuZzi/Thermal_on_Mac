"""Local fragment-pair hypotheses and a separate fixed-C rematching control.

Two measured fragments constrain a short interpolation. Continuous NMS support
inside the gap is NOT required. Original C weights/thresholds and subsequent R
stay unchanged, and every old accepted path is preserved. This is a proposal
experiment, not learned completion or a guarantee of boundary ownership.
"""

import copy
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from .clarity_semantic_edges import normal_ridges
from .completion_recall import _score, _validate_source
from .contour_completion import (
    _STEPS, _diagonals, _find_path, _path_features, _render_paths, evidence_fields,
)
from .semantic_edges import _unit_map


@dataclass(frozen=True)
class FragmentSettings:
    minimum_fragment_fraction: float = 3 / 288
    minimum_tangent_quality: float = .65
    minimum_scale_agreement: float = 2 / 3
    region_cost_weight: float = .5
    max_anchors: int = 8192
    max_searches: int = 512

    def __post_init__(self):
        for name in ("minimum_fragment_fraction", "minimum_tangent_quality", "minimum_scale_agreement", "region_cost_weight"):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1:
                raise ValueError(f"Finite fragment fraction inside (0,1] required: {name}")
        if self.minimum_fragment_fraction>.15:raise ValueError("Local fragments required")
        for name in ("max_anchors","max_searches"):
            v=getattr(self,name)
            if type(v) is not int or v<1:raise ValueError(f"Positive integer budget required: {name}")


def _graph(ridges):
    """8-neighbors, excluding redundant diagonals around an orthogonal corner."""
    pixels={tuple(p) for p in np.argwhere(ridges)}
    graph={}
    for y,x in sorted(pixels):
        following=[]
        for dy,dx in _STEPS:
            dy,dx=int(dy),int(dx)
            q=(y+dy,x+dx)
            if q not in pixels:continue
            if dy and dx and ((y+dy,x) in pixels or (y,x+dx) in pixels):continue
            following.append(q)
        graph[y,x]=sorted(following)
    return graph


def _segments(ridges):
    """Decompose measured ridges at tips/junctions; isolated loops have no tips."""
    graph=_graph(ridges);visited=set();segments=[]
    for p in sorted(graph):
        if len(graph[p])==2:continue
        for q in graph[p]:
            edge=tuple(sorted((p,q)))
            if edge in visited:continue
            path=[p,q];visited.add(edge)
            while len(graph[path[-1]])==2:
                candidates=[v for v in graph[path[-1]] if v!=path[-2]]
                next_point=candidates[0]
                edge=tuple(sorted((path[-1],next_point)))
                if edge in visited:break
                visited.add(edge);path.append(next_point)
            segments.append(path)
    return segments,graph


def _ports(ridges, observed, fields, smooth, source, settings, original):
    segments,graph=_segments(ridges)
    minimum=max(3,int(math.ceil(min(ridges.shape)*settings.minimum_fragment_fraction)))
    window=max(minimum,int(math.ceil(min(ridges.shape)*original.tangent_fraction)))
    offset=max(1.,2*min(ridges.shape)/288)
    ports=[];reasons=Counter()
    for segment_id,segment in enumerate(segments):
        if len(segment)<minimum:
            reasons['short_fragment']+=1;continue
        for chain in (segment[:window],list(reversed(segment))[:window]):
            points=np.asarray(chain,dtype=np.float64);start=tuple(int(v) for v in points[0])
            if min(start)<=0 or start[0]>=ridges.shape[0]-1 or start[1]>=ridges.shape[1]-1:
                reasons['image_border']+=1;continue
            values,vectors=np.linalg.eigh(np.cov(points,rowvar=False,bias=True))
            quality=float((values[-1]-values[0])/max(values.sum(),1e-12))
            if quality<settings.minimum_tangent_quality:
                reasons['direction_uncertain']+=1;continue
            tangent=vectors[:,-1]
            if tangent@(points[0]-points[-1])<0:tangent=-tangent
            normal=np.array([-tangent[1],tangent[0]])
            yy,xx=points.astype(int).T
            directional=np.abs(normal[0]*fields['vectors'][:,0,yy,xx]+normal[1]*fields['vectors'][:,1,yy,xx])
            support=float(directional.mean())
            if support<original.geometry_min_support:
                reasons['anchor_image_conflict']+=1;continue
            inherited=float(np.median(observed[yy,xx]))
            if source=='observed' and inherited<original.trace_threshold:
                reasons['unsupported_observed_fragment']+=1;continue
            left,right=points+offset*normal,points-offset*normal
            inside=lambda p:bool(((p>=0)&(p<=np.asarray(ridges.shape)-1)).all())
            context_valid=inside(left) and inside(right)
            sides=[float(map_coordinates(smooth,p.T,order=1,mode='nearest').mean()) for p in (left,right)]
            ports.append(dict(point=list(start),tangent=tangent.tolist(),tangent_quality=quality,
                strength=inherited,observation_strength=support,anchor_support=support,
                trace_length=len(chain),trace_yx=[[int(v) for v in p] for p in chain],
                source=source,segment_id=f'{source}:{segment_id}',junction=len(graph[start])>2,
                side_means=sides,context_valid=context_valid))
    return ports,dict(reasons),len(segments)


def _region_cost(a,b,noise):
    if not a['context_valid'] or not b['context_valid']:return 0.
    # B's outward tangent points opposite to travel A -> B, so swap its sides.
    first=np.asarray(a['side_means']);second=np.asarray(b['side_means'])[::-1]
    scale=abs(first[0]-first[1])+abs(second[0]-second[1])+2*noise+1/255
    return float(np.clip(np.abs(first-second).mean()/scale,0,1))


def _old_locks(metadata,shape,accepted_only=False):
    occupied=set();painted=np.zeros(shape,bool);diagonals={}
    for c in metadata['candidates']:
        if not c['geometry_selected'] or (accepted_only and not c['reliability_accepted']):continue
        path=np.asarray(c['path_yx'],dtype=int)
        occupied.update(tuple(p) for p in path[[0,-1]])
        painted[tuple(path.T)]=True;diagonals.update(_diagonals(path))
    return occupied,painted,diagonals


def _conflict(path,occupied,painted,diagonals):
    if any(tuple(p) in occupied for p in path[[0,-1]]):return 'endpoint_already_used'
    diagonal=_diagonals(path)
    if painted[tuple(path.T)].any() or any(cell in diagonals and diagonals[cell]!=slope for cell,slope in diagonal.items()):
        return 'candidate_crossing'
    return None


def _occupy(path,occupied,painted,diagonals):
    occupied.update(tuple(p) for p in path[[0,-1]])
    painted[tuple(path.T)]=True;diagonals.update(_diagonals(path))


def rematch_fixed_C(image,observed,ridges,metadata):
    """Release rejected geometry, but NEVER rescore C or remove accepted paths."""
    image=_unit_map(image,'TIR');observed=_unit_map(observed,'observed',image.shape)
    ridges=np.asarray(ridges)
    if ridges.shape!=image.shape or ridges.dtype!=np.bool_:raise ValueError('Aligned boolean ridges required')
    settings=_validate_source(metadata,ridges)
    updated=copy.deepcopy(metadata)
    occupied,painted,diagonals=_old_locks(metadata,image.shape,accepted_only=True)
    added=[]
    for c in sorted(updated['candidates'],key=lambda c:(c['cost'],c['id'])):
        if c['reliability_accepted']:continue
        c['geometry_selected']=False
        if c['reliability_rejections']:
            c['geometry_rejection']='C_rejected_before_matching';continue
        path=np.asarray(c['path_yx'],dtype=int)
        conflict=_conflict(path,occupied,painted,diagonals)
        if conflict:
            c['geometry_rejection']=conflict;continue
        c.update(geometry_selected=True,reliability_accepted=True,geometry_rejection=None)
        _occupy(path,occupied,painted,diagonals);added.append(c['id'])
    updated['geometry_selected']=sum(c['geometry_selected'] for c in updated['candidates'])
    updated['reliability_accepted']=sum(c['reliability_accepted'] for c in updated['candidates'])
    updated['rematch']=dict(format='fixed_C_rematch_v1',new_accepted_ids=added,
        C_rescored=False,original_accepted_preserved=True,
        limits='Original alternative-margin C may already reject backups; this isolates matching only')
    _,base=_render_paths(image.shape,metadata['candidates'],settings.soft_width)
    _,new=_render_paths(image.shape,updated['candidates'],settings.soft_width)
    if (new<base).any():raise AssertionError('Old accepted paths must survive rematching')
    return dict(baseline=np.maximum(observed,base),rematched=np.maximum(observed,new)),updated


def _path_rejection(path,ridges):
    """A boundary hypothesis cannot revisit a grid point or an anchor inside."""
    if len(np.unique(path,axis=0))!=len(path):return 'non_simple_path'
    if ridges[tuple(path[1:-1].T)].any():return 'observed_boundary_interior'
    return None


def _finalize_pool(updated,metadata,shape,original):
    """Common deterministic C/matching pass for fresh and cached searches."""
    n=len(metadata['candidates']);new=updated['candidates'][n:]
    by_site={}
    for c in updated['candidates']:
        for i in c['endpoints']:
            site=tuple(updated['endpoints'][i]['point'])
            by_site.setdefault(site,[]).append((c['id'],c['cost']))
    for c in new:
        c.update(geometry_selected=False,reliability_accepted=False,geometry_rejection=None,reliability_rejections=[])
        alternatives=[cost for i in c['endpoints'] for j,cost in by_site[tuple(updated['endpoints'][i]['point'])] if j!=c['id']]
        _score(c,alternatives,original)
    occupied,painted,diagonals=_old_locks(metadata,shape)
    for c in sorted(new,key=lambda c:(c['cost'],c['id'])):
        path=np.asarray(c['path_yx'],dtype=int)
        conflict=_conflict(path,occupied,painted,diagonals)
        if conflict:c['geometry_rejection']=conflict;continue
        c['geometry_selected']=True;c['reliability_accepted']=not c['reliability_rejections']
        _occupy(path,occupied,painted,diagonals)


def sanitize_cached_fragments(image,observed,ridges,metadata,cached):
    """Replay a frozen search pool through the corrected path admissibility.

    Search geometry/features are data, not instructions. Caller verifies input
    and cached-recipe hashes. R is path/context-only and can remain cached when
    C is recomputed; a fresh-frame equivalence check verifies this shortcut.
    """
    image=_unit_map(image,'TIR');observed=_unit_map(observed,'observed',image.shape)
    ridges=np.asarray(ridges)
    if ridges.dtype!=np.bool_ or ridges.shape!=image.shape:raise ValueError('Aligned boolean source ridges required')
    original=_validate_source(metadata,ridges)
    n=len(metadata['candidates']);details=cached.get('fragment_completion',{})
    if details.get('original_candidates')!=n or cached['candidates'][:n]!=metadata['candidates']:
        raise ValueError('Cached and original candidate identities differ')
    updated=copy.deepcopy(cached);updated['candidates']=copy.deepcopy(metadata['candidates'])
    reasons=Counter();discarded=[];retained=[]
    from .completion_importance import _validate_candidate
    for candidate in cached['candidates'][n:]:
        path=_validate_candidate(candidate,image.shape)
        reason=_path_rejection(path,ridges)
        if reason:
            reasons[reason]+=1;discarded.append(candidate['id']);continue
        c=copy.deepcopy(candidate);retained.append(c['id']);c['id']=len(updated['candidates'])
        updated['candidates'].append(c)
    _finalize_pool(updated,metadata,image.shape,original)
    new=updated['candidates'][n:];d=updated['fragment_completion']
    d.update(format='two_fragment_completion_v2',new_candidates=len(new),
        new_geometry_selected=sum(c['geometry_selected'] for c in new),
        new_reliability_accepted=sum(c['reliability_accepted'] for c in new),
        path_admissibility='Unique grid points; no original ridge at interior pixels')
    d['pair_rejections']=dict(Counter(d['pair_rejections'])+reasons)
    d['cached_search_replay']=dict(discarded_ids=discarded,retained_original_ids=retained,
        rejection_counts=dict(reasons),C_and_matching_recomputed=True,R_depends_on_unchanged_path_and_context=True)
    return updated


def fragment_completion(image,observed,ridges,metadata,*,settings=None):
    """Append local two-fragment proposals; defer missing-pixel support to C.

    Returns inherited-E amplitude and a SEPARATE observation-amplitude probe on
    identical paths/C. The latter must not be called more accurate for being
    brighter. Region compatibility is a bounded soft cost, never an identity
    label or a semantic hard veto.
    """
    settings=FragmentSettings() if settings is None else settings
    if not isinstance(settings,FragmentSettings):raise TypeError('Use validated FragmentSettings')
    image=_unit_map(image,'TIR');observed=_unit_map(observed,'observed',image.shape)
    ridges=np.asarray(ridges)
    if ridges.shape!=image.shape or ridges.dtype!=np.bool_ or min(image.shape)<8:raise ValueError('Aligned boolean original ridges required')
    if 'fragment_completion' in metadata:raise ValueError('Do not append fragments twice')
    original=_validate_source(metadata,ridges)
    fields=evidence_fields(image)
    strengths=np.linalg.norm(fields['vectors'],axis=1)
    weak=normal_ridges(image,fields['strength']) & (fields['strength']>=original.support_threshold)
    weak&=(strengths>=original.support_threshold).mean(0)+1e-12>=settings.minimum_scale_agreement
    smooth=gaussian_filter(image,max(.45,min(image.shape)/288),mode='reflect')
    observed_ports,obs_reasons,obs_segments=_ports(ridges,observed,fields,smooth,'observed',settings,original)
    weak_ports,weak_reasons,weak_segments=_ports(weak,observed,fields,smooth,'weak',settings,original)
    ports=observed_ports.copy();duplicate=0
    tree=cKDTree([p['point'] for p in ports]) if ports else None
    duplicate_radius=max(1.,min(image.shape)/288)
    for p in weak_ports:
        neighbors=tree.query_ball_point(p['point'],duplicate_radius) if tree is not None else []
        if any(np.dot(p['tangent'],ports[i]['tangent'])>.95 for i in neighbors):duplicate+=1;continue
        ports.append(p)
    # A class-independent bound, with truncation recorded rather than hidden.
    ports.sort(key=lambda p:(-p['tangent_quality']*p['anchor_support'],p['point'],p['tangent'],p['source']))
    truncated_anchors=max(0,len(ports)-settings.max_anchors);ports=ports[:settings.max_anchors]
    updated=copy.deepcopy(metadata);old_count=len(updated['candidates'])
    old_endpoint_count=len(updated['endpoints'])
    occupied,painted,diagonals=_old_locks(metadata,image.shape)
    blocked=ridges|painted
    free=[]
    for p in ports:
        if tuple(p['point']) in occupied:continue
        p=dict(p,id=len(updated['endpoints']))
        updated['endpoints'].append(p);free.append(p)
    maximum=metadata['gap_limit_pixels'];cosine=math.cos(math.radians(original.max_endpoint_angle))
    pairs=[];reasons=Counter();expanded=0
    if free:
        coordinates=np.asarray([p['point'] for p in free]);tree=cKDTree(coordinates)
        neighbors=[]
        for i,p in enumerate(coordinates):
            local=sorted((j for j in tree.query_ball_point(p,maximum) if j!=i),key=lambda j:(float(np.linalg.norm(p-coordinates[j])),j))
            # Co-located junction ports do not use up the nearby-fragment budget.
            local=[j for j in local if np.linalg.norm(p-coordinates[j])>math.sqrt(2)+1e-6]
            compatible=[]
            for j in local:
                delta=coordinates[j]-p;direction=delta/np.linalg.norm(delta)
                if (free[i]['segment_id']!=free[j]['segment_id']
                    and direction@free[i]['tangent']>=cosine
                    and -direction@free[j]['tangent']>=cosine):compatible.append(j)
            local=compatible
            neighbors.append(set(local[:original.max_neighbors]))
        for a,b in sorted(tree.query_pairs(maximum)):
            first,second=free[a],free[b]
            if first['segment_id']==second['segment_id']:
                reasons['same_measured_segment']+=1;continue
            delta=coordinates[b]-coordinates[a];gap=float(np.linalg.norm(delta))
            if gap<=math.sqrt(2)+1e-6:
                reasons['adjacent']+=1;continue
            alignment=(float(delta/gap@first['tangent']),float(-delta/gap@second['tangent']))
            if min(alignment)<cosine:
                reasons['fragment_direction']+=1;continue
            if b not in neighbors[a] and a not in neighbors[b]:
                reasons['neighbor_budget']+=1;continue
            region=_region_cost(first,second,fields['noise_sigma'])
            priority=gap/maximum+.5*(2-sum(alignment))+settings.region_cost_weight*region
            pairs.append((priority,gap,a,b,region))
    pairs.sort();truncated_pairs=max(0,len(pairs)-settings.max_searches)
    for _,gap,a,b,region in pairs[:settings.max_searches]:
        first,second=free[a],free[b]
        # The original search uses tuple coordinates and NumPy directions.
        # Public/cached fragment records remain JSON-compatible lists.
        first_search=dict(first,point=tuple(first['point']),tangent=np.asarray(first['tangent']))
        second_search=dict(second,point=tuple(second['point']),tangent=np.asarray(second['tangent']))
        result,reason,states=_find_path(first_search,second_search,blocked,fields,original,gap);expanded+=states
        if result is None:reasons[reason]+=1;continue
        path,path_cost,length=result
        reason=_path_rejection(path,ridges)
        if reason:reasons[reason]+=1;continue
        features=_path_features(path,first,second,fields,original)
        inherited=min(first['strength'],second['strength'])
        observation=min(first['observation_strength'],second['observation_strength'])
        # Do not pre-reject a gap for low mean image support. Record its C
        # rejection instead: this is the key difference from edge following.
        updated['candidates'].append(dict(id=len(updated['candidates']),endpoints=[first['id'],second['id']],
            path_yx=path.tolist(),gap=gap,length=length,cost=float(path_cost+settings.region_cost_weight*region),
            strength=float(inherited),observation_amplitude=float(observation),
            region_disagreement=region,path_cost_before_region=float(path_cost),
            features=features,geometry_selected=False,reliability_accepted=False,
            geometry_rejection=None,reliability_rejections=[],source='two_measured_fragments',
            anchor_sources=[first['source'],second['source']],junction_anchor=first['junction'] or second['junction']))
    _finalize_pool(updated,metadata,image.shape,original)
    new=updated['candidates'][old_count:]
    probe=copy.deepcopy(updated)
    for c in probe['candidates'][old_count:]:c['strength']=c['observation_amplitude']
    updated['fragment_completion']=dict(format='two_fragment_completion_v2',settings=asdict(settings),
        original_candidates=old_count,original_endpoints=old_endpoint_count,
        observed_segments=obs_segments,weak_segments=weak_segments,
        observed_ports=len(observed_ports),weak_ports=len(weak_ports),duplicate_weak_ports=duplicate,
        admitted_ports=len(free),truncated_anchors=truncated_anchors,eligible_pairs=len(pairs),truncated_pairs=truncated_pairs,
        pair_rejections=dict(reasons),observed_port_rejections=obs_reasons,weak_port_rejections=weak_reasons,
        new_candidates=len(new),new_geometry_selected=sum(c['geometry_selected'] for c in new),
        new_reliability_accepted=sum(c['reliability_accepted'] for c in new),expanded_states=expanded,
        matching='Original geometry locks preserved; separate fixed-C rematch is another arm',
        region='Class-independent TIR flank similarity, bounded soft cost; not instance identity',
        confidence='Unchanged original C formula; candidate interpolation can still be refused',
        amplitude_control='Inherited E vs measured fragment support; same paths, geometry, C and R',
        path_admissibility='Unique grid points; no original ridge at interior pixels')
    probe['fragment_completion']=copy.deepcopy(updated['fragment_completion'])
    probe['fragment_completion']['encoding']='observation_amplitude_probe'
    return updated,probe,dict(weak_ridges=weak)
