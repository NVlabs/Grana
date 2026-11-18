# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grana.inference.selection import select_prediction
from scripts.inference import benchmark_batch as benchmark


@pytest.fixture
def campaign(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'metadata.json').write_text('{"fixture": true}\n')
    plan = {'schema': 'grana-http-frame-comparison-plan-v1', 'seeds': [11, 29],
            'samples_per_trajectory': 8, 'stride_pair': [16, 4],
            'checkpoint': {'path': str(checkpoint), 'weights_sha256': 'a' * 64}, 'trajectories': []}
    for index in range(24):
        path = tmp_path / f'episode_{index}.npy'
        np.save(path, np.arange(80 * 2 * 3 * 3, dtype=np.uint8).reshape(80, 2, 3, 3))
        request = tmp_path / f'request_{index}.json'
        request.write_text(json.dumps({'views': {'top': path.name}, 'task': 'Assemble',
                                       'stage_prompts': ['Connect', 'Finish']}))
        plan['trajectories'].append({'key': f'key_{index}', 'trajectory_id': f'trajectory_{index}',
            'group': 'L1L2' if index < 12 else 'L3L4', 'num_frames': 80, 'stage_count': 2, 'source_fps': 10,
            'anchors': [0, 79], 'targets': {'11': list(range(2, 73, 10)), '29': list(range(3, 74, 10))},
            'request_path': request.name, 'metadata_path': 'MUST_NOT_READ_GROUND_TRUTH.json'})
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    args = SimpleNamespace(plan=plan_path, input=None, checkpoint=None, seed=None, batch_size=64,
                           prediction_frame='poi', roles=['C', 'D'], warmup=2, repeats=5,
                           model_name_or_path=None, output=tmp_path / 'benchmark.json')
    return args, plan


class FakeCuda:
    def __init__(self, visible=1):
        self.visible, self.elapsed, self.peak = visible, 0., 0
        self.events = []

    def device_count(self):
        return self.visible

    def set_device(self, device):
        assert device == 0
        self.events.append('set_device')

    def synchronize(self, device):
        assert device == 0
        self.events.append('sync')

    def reset_peak_memory_stats(self, device):
        assert device == 0
        self.peak = 100
        self.events.append('reset_peak')

    def max_memory_allocated(self, device):
        return self.peak

    def max_memory_reserved(self, device):
        return self.peak + 50

    def get_device_properties(self, device):
        return SimpleNamespace(name='Fake H100', uuid='fake-uuid', total_memory=100_000)


class FakePredictor:
    def __init__(self, cuda, *, invalid_focus=False):
        self.cuda, self.invalid_focus, self.calls, self.seen = cuda, invalid_focus, [], set()
        self.counter = 0
        self.checkpoint = {'weights_sha256': 'a' * 64,
                           'inference_runtime': {'policy': 'cuda_optimized_v1', 'gpu_image_resize': True}}

    def runtime_counters(self):
        return {'decoder_compile': {'compiled_calls': self.counter, 'fallbacks': 0}}

    def response(self, request, batch):
        # Batch predictions deliberately differ; comparisons must not gate speed.
        ids = [(frame + int(batch)) % 2 for frame in request.frame_indices]
        focus = ids[request.poi_slot]
        values = [.2 + .05 * int(batch)] * 8
        if self.invalid_focus:
            focus = (focus + 1) % 2
        result = {'checkpoint': deepcopy(self.checkpoint), 'roles': ['C', 'D'], 'poi_slot': request.poi_slot,
                  'prediction_frame': request.prediction_frame, 'frame_indices': list(request.frame_indices),
                  'stage': {'ids': ids}, 'D_focus_stage': focus,
                  'progress': {'D': {'values': values, 'focus_stage': focus}}}
        result['selected_prediction'] = select_prediction(result)
        return result

    def predict_batch(self, requests, roles, batch_size):
        assert roles == ('C', 'D') and batch_size == len(requests)
        self.calls.append(('batch', [id(request) for request in requests]))
        self.cuda.events.append('batch')
        self.cuda.elapsed += .002 * len(requests) + (5 if 'batch' not in self.seen else 0)
        self.seen.add('batch')
        self.cuda.peak = max(self.cuda.peak, 100 + 10 * len(requests))
        self.counter += 2
        return [self.response(request, True) for request in requests]

    def predict(self, request, roles):
        assert roles == ('C', 'D')
        self.calls.append(('serialized', id(request)))
        self.cuda.events.append('predict')
        self.cuda.elapsed += .01 + (5 if 'serialized' not in self.seen else 0)
        self.seen.add('serialized')
        self.cuda.peak = max(self.cuda.peak, 110)
        self.counter += 2
        return self.response(request, False)


def execute(args, monkeypatch, *, invalid_focus=False, visible=1):
    cuda = FakeCuda(visible)
    predictor = FakePredictor(cuda, invalid_focus=invalid_focus)
    monkeypatch.setattr(benchmark, 'source_identity', lambda: {'sha256': {'fixture.py': 'b' * 64}})
    def factory(path, model_name_or_path, device):
        assert cuda.events == ['set_device'] and device == 'cuda:0'
        return predictor
    report = benchmark.run(args, predictor_factory=factory, cuda=cuda, clock=lambda: cuda.elapsed)
    return report, predictor, cuda


def test_plan_workload_has_64_distinct_real_quality_windows_covering_all_24(campaign):
    args, plan = campaign
    requests, workload, checkpoint, weights = benchmark.prepare_workload(args)
    assert len(requests) == workload['count'] == 64
    assert len(workload['trajectory_counts']) == 24
    assert Counter(workload['trajectory_counts'].values()) == {3: 16, 2: 8}
    assert len({(item['key'], item['source_frame']) for item in workload['windows']}) == 64
    assert all(item['source_frame'] in plan['trajectories'][int(item['key'][4:])]['targets']['11']
               for item in workload['windows'])
    assert all(item['kind'] == 'quality' and item['seed'] == 11 for item in workload['windows'])
    assert all(request.frame_indices[5] == item['source_frame'] for request, item in zip(requests, workload['windows']))
    assert len(workload['input_sha256']) == 64 and weights == 'a' * 64
    assert checkpoint.is_absolute()
    assert benchmark.prepare_workload(args)[1] == workload


def test_small_workload_does_not_request_zero_samples_from_other_trajectories(campaign):
    args, _ = campaign
    args.batch_size = 3
    requests, workload, _, _ = benchmark.prepare_workload(args)
    assert len(requests) == 3 and len(workload['trajectory_counts']) == 3


def test_last_mode_is_future_free_and_hashes_the_actual_window(campaign):
    args, _ = campaign
    poi = benchmark.prepare_workload(args)[1]
    args.prediction_frame = 'last'
    requests, last, _, _ = benchmark.prepare_workload(args)
    assert last['input_sha256'] != poi['input_sha256']
    assert all(request.poi_slot == 7 and max(request.frame_indices) <= item['source_frame']
               for request, item in zip(requests, last['windows']))


def test_plan_duplicate_source_aliases_cannot_supply_fake_distinct_windows(campaign):
    args, plan = campaign
    plan['trajectories'] = plan['trajectories'][:2]
    plan['trajectories'][1]['request_path'] = plan['trajectories'][0]['request_path']
    args.plan.write_text(json.dumps(plan))
    args.batch_size = 12
    with pytest.raises(ValueError, match='no windows are duplicated'):
        benchmark.prepare_workload(args)


def test_full_video_input_support_and_insufficient_real_frames(campaign):
    args, plan = campaign
    args.input = args.plan.parent / plan['trajectories'][0]['request_path']
    args.plan, args.checkpoint = None, args.plan.parent / 'checkpoint'
    requests, workload, _, _ = benchmark.prepare_workload(args)
    assert len(requests) == 64 and len({item['source_frame'] for item in workload['windows']}) == 64
    args.batch_size = 81
    with pytest.raises(ValueError, match='Insufficient distinct'):
        benchmark.prepare_workload(args)


def test_whole_public_calls_are_timed_and_divided_by_64_only(campaign, monkeypatch):
    args, _ = campaign
    report, predictor, cuda = execute(args, monkeypatch)
    assert report['execution'] == 'local_public_predictor' and report['transport'] == 'local'
    assert report['timing']['amortization_divisor'] == 64
    assert report['timing']['no_division_by_frame_or_camera_count'] is True
    assert report['variants']['batch']['batch_latency_ms']['mean'] == pytest.approx(128.)
    assert report['variants']['batch']['per_selected_poi_amortized_ms']['mean'] == pytest.approx(2.)
    assert report['variants']['serialized']['batch_latency_ms']['mean'] == pytest.approx(640.)
    assert report['variants']['serialized']['per_selected_poi_amortized_ms']['mean'] == pytest.approx(10.)
    assert report['throughput_speedup_batch_over_serialized'] == pytest.approx(5.)
    assert report['variants']['batch']['peak_allocated_bytes'] == 740
    assert report['variants']['serialized']['peak_allocated_bytes'] == 110
    assert len(report['raw_rounds']) == 14
    assert report['raw_rounds'][0]['batch_latency_ms'] == pytest.approx(5128.)
    batch_calls = [request_ids for variant, request_ids in predictor.calls if variant == 'batch']
    serial_calls = [request_id for variant, request_id in predictor.calls if variant == 'serialized']
    assert len(batch_calls) == 7 and len(serial_calls) == 7 * 64
    assert all(ids == batch_calls[0] for ids in batch_calls)
    assert serial_calls == batch_calls[0] * 7
    for pair in range(7):
        rows = report['raw_rounds'][2 * pair:2 * pair + 2]
        assert [row['variant'] for row in rows] == list(benchmark.round_order(pair))
    assert cuda.events.count('sync') == 28
    assert json.loads(args.output.read_text()) == report


def test_validation_is_outside_timing_and_no_output_parity_gate(campaign, monkeypatch):
    args, _ = campaign
    args.batch_size, args.warmup, args.repeats = 4, 1, 2
    cuda, original = FakeCuda(), benchmark.validate_reply
    predictor = FakePredictor(cuda)
    def validate(*a, **k):
        cuda.elapsed += 100.
        return original(*a, **k)
    monkeypatch.setattr(benchmark, 'validate_reply', validate)
    monkeypatch.setattr(benchmark, 'source_identity', lambda: {})
    report = benchmark.run(args, predictor_factory=lambda *a, **k: predictor, cuda=cuda, clock=lambda: cuda.elapsed)
    assert report['variants']['batch']['per_selected_poi_amortized_ms']['mean'] == pytest.approx(2.)
    assert report['variants']['serialized']['per_selected_poi_amortized_ms']['mean'] == pytest.approx(10.)


def test_invalid_focus_fails_without_publishing_report(campaign, monkeypatch):
    args, _ = campaign
    args.batch_size = 2
    with pytest.raises(ValueError, match='D focus'):
        execute(args, monkeypatch, invalid_focus=True)
    assert not args.output.exists()


def test_requires_isolated_gpu_and_preserves_existing_report(campaign, monkeypatch):
    args, _ = campaign
    args.batch_size = 2
    with pytest.raises(RuntimeError, match='Isolate one GPU'):
        execute(args, monkeypatch, visible=8)
    args.output.write_text('keep')
    with pytest.raises(FileExistsError):
        execute(args, monkeypatch)
    assert args.output.read_text() == 'keep'


def test_source_identity_distinguishes_dirty_checkout_from_recorded_head(monkeypatch):
    def git(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='abc123\n' if command[-1] == 'HEAD' else '?? benchmark_batch.py\n')
    monkeypatch.setattr(benchmark.subprocess, 'run', git)
    monkeypatch.setattr(benchmark, 'file_digest', lambda path: 'b' * 64)
    identity = benchmark.source_identity()
    assert identity['git_commit'] == 'abc123' and identity['git_dirty'] is True
    assert set(identity['sha256']) == set(benchmark.SOURCE_FILES)
