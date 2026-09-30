import unittest
import csv
import json
from pathlib import Path
import tempfile

from benchmarks.local_llm import NoRedirects, cases, local_model, make_prompt, parse_advice
from benchmarks.local_evaluate import evaluate


class LocalAdviceTests(unittest.TestCase):
    def test_parse_preserves_model_coordinates(self):
        self.assertEqual(parse_advice('Generated Path: [[2, 3], [99, -2]]'),
                         ([[2, 3], [99, -2]], 'ok'))
        self.assertEqual(parse_advice('```json\n[[2,3],[4,5]]\n```')[0], [[2,3],[4,5]])

    def test_malformed_advice_has_no_repair(self):
        for text in ['no route', '[[2,3], [4,', '[[2.5,3]]', '[[True,3]]',
                     '[[2,3,4]]', "[[__import__('os').getcwd(), 3]]"]:
            points, status = parse_advice(text)
            self.assertEqual(points, [])
            self.assertNotEqual(status, 'ok')

    def test_case_selection_is_fixed_without_oracle(self):
        selected = cases('test')
        self.assertEqual(len(selected), 120)
        self.assertEqual(len({r['map_id'] for r in selected}), 60)
        self.assertEqual({r['sample_id'] for r in selected}, {0,5})
        q = selected[0]['query']
        t = cases('test', max_maps=1, sample_ids=(0,), domain='transpose')[0]['query']
        self.assertEqual(t['start'], q['start'][::-1])
        self.assertEqual(t['horizontal_barriers'], q['vertical_barriers'])
        s = cases('test', max_maps=1, sample_ids=(0,), domain='scale2')[0]['query']
        self.assertEqual(s['start'], [2*x for x in q['start']])

    def test_prompt_contains_only_query_and_fixed_examples(self):
        q = cases('test')[0]['query']
        prompt = make_prompt(q)
        self.assertIn('Grid bounds:', prompt)
        self.assertTrue(prompt.endswith('Generated Path: \n'))
        self.assertIn(str(q['goal']), prompt)

    def test_unapproved_or_cloud_model_rejected_before_network(self):
        for name in ['gpt-4', 'qwen3:cloud', 'llama3.2:3b-cloud']:
            with self.assertRaises(ValueError):
                local_model(name)

    def test_local_api_never_follows_remote_redirect(self):
        with self.assertRaises(ValueError):
            NoRedirects().redirect_request(None, None, 302, '', {}, 'https://example.com')

    def test_generation_failure_is_kept_in_all_comparisons(self):
        record = cases('validation', max_maps=1, sample_ids=(0,))[0]
        record.update(waypoints=[], parse_status='missing_list',
                      generation={'done_reason': 'stop'}, provenance={'digest': 'unit-test-fixture'})
        with tempfile.TemporaryDirectory() as folder:
            evaluate([record], Path(folder))
            with (Path(folder)/'raw.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows),25)
            self.assertTrue(all(r['success']=='True' for r in rows))
            quality = json.loads((Path(folder)/'quality.json').read_text())
            self.assertTrue(quality[0]['no_intermediate_advice'])
            self.assertEqual(quality[0]['parse_status'],'missing_list')


if __name__ == '__main__':
    unittest.main()
