"""Validate actual role isolation and copied-image leakage across modalities."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.grouped_unaligned_dataset import validate_manifest,validate_prior_partitions


class GroupedDataTests(unittest.TestCase):
    def test_video_role_conflicts_and_reencoded_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pixels=np.zeros((19,31,3),np.uint8); pixels[3:8]=200
            Image.fromarray(pixels).save(root/'rgb.png'); Image.fromarray(pixels).save(root/'copied.png')
            path=root/'groups.json'
            data=dict(format='grouped_unaligned_v1',domains={
                'A':[dict(image='rgb.png',group='video_1',split='train')],
                'B':[dict(image='copied.png',group='video_1',split='val')]})
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'group leaks'): validate_manifest(path)
            data['domains']['B'][0]['group']='another_video'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'duplicate image leaks'): validate_manifest(path)
            pixels[0,0]=90; Image.fromarray(pixels).save(root/'copied.png')
            manifest,_=validate_manifest(path)
            self.assertEqual(manifest['domains']['B'][0]['split'],'val')

    def test_prior_cannot_train_on_gan_validation_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prior=root/'prior.json'
            prior.write_text(json.dumps(dict(provenance=dict(dataset_id='sample',samples=[
                dict(group='holdout',split='train',rgb_content_sha256='unused')]))))
            manifest=dict(dataset_id='sample',domains={'A':[], 'B':[dict(group='holdout',split='val',image='not_read')]})
            with self.assertRaisesRegex(ValueError,'fitting leaks'):
                validate_prior_partitions(manifest,root,prior)


if __name__=='__main__': unittest.main()
