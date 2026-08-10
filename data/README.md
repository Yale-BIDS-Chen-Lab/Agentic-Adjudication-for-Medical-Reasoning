# Data

Benchmark data is not included.

Prepare datasets locally using the JSONL schema below and pass each file with
`--data-path`. The runners also accept `--dataset` and `--split` overrides.

## MCQ Schema

Each line is one JSON object:

```json
{
  "realidx": "0",
  "question": "A patient ... What is the most likely diagnosis?",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "answer_idx": "A"
}
```

Accepted answer labels can be single-label (`"A"`) or comma-separated
multi-label (`"A,C"`).

## Expected Layout

The code does not require this exact layout, but these paths match the examples:

```text
data/
  MCQ/
    medqa/
      test.jsonl
      test_hard.jsonl
      sampled_50.jsonl
    pubmedqa/
      test.jsonl
      test_hard.jsonl
      sampled_50.jsonl
    NEJM/
      internal_medicine/
        test.jsonl
```

For HealthBench and MedRBench, use the official benchmark repositories and pass
the local file path to the corresponding runner.
