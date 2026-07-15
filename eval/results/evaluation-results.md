# EvidenceFlow Evaluation Results

Generated: `2026-07-14T23:11:16.055625+00:00`  
Bundles: **20**  
Duration: **3971.48s**

## Reproducibility identity

| Identity | Value |
| --- | --- |
| Classification model | `gemma4:12b-mlx` |
| Classification model digest | `197a75677efb4b634352a8bdf24fd4781f1ea2c0b0c11f2b391ea7e0fcdcf01c` |
| Extraction model | `gemma4:12b-mlx` |
| Extraction model digest | `197a75677efb4b634352a8bdf24fd4781f1ea2c0b0c11f2b391ea7e0fcdcf01c` |
| Reporting model | `gemma4:12b-mlx` |
| Reporting model digest | `197a75677efb4b634352a8bdf24fd4781f1ea2c0b0c11f2b391ea7e0fcdcf01c` |
| Embedding model | `embeddinggemma` |
| Embedding model digest | `85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` |
| Dataset SHA-256 | `4d2dbef7a5821d12e178c5387c196eb4f58da05de4e71fa4b6f4affc1c6e47d9` |
| Configuration SHA-256 | `556991efbcec38c779fb2a1f53ef04938fbcc8c11f14a3dd676e203c630ac728` |
| Policy-query labels SHA-256 | `d397b93bb60c66f1f0f6d557310885d9d29aba611ca58c42c2fd3acee74cddb8` |
| Python | `3.12.13` |
| cache_enabled | `True` |
| cache_hit_count | `0` |
| cache_schema_version | `1` |
| implementation_sha256 | `473de267f8a8cf8d6003cff3cb8e8d0525d4f1964592a47dbd8ca18ca8ca5d69` |

## Quality metrics

| Metric | Result |
| --- | ---: |
| Document classification accuracy | 1.0000 |
| Field extraction accuracy | 0.9953 |
| Missing-document detection accuracy | 1.0000 |
| Conflict precision | 1.0000 |
| Conflict recall | 1.0000 |
| Human-review routing accuracy | 0.8500 |
| Human-review required accuracy | 0.8500 |
| Report citation validity | 1.0000 |
| Report unknown-ID rate | 0.0000 |
| Policy HitRate@5 | 1.0000 |
| Policy Recall@5 | 0.7500 |
| Policy MRR@5 | 0.8750 |
| Policy nDCG@5 | 0.7246 |

## Counts

| Count | Value |
| --- | ---: |
| Conflict False Negative | 0 |
| Conflict False Positive | 0 |
| Conflict True Positive | 6 |
| Documents | 64 |
| Labelled Fields | 215 |
| Report Citations | 75 |
| Review Routes | 20 |

## Latency

| Measure | Value |
| --- | ---: |
| Minimum bundle latency | 160.92s |
| Mean bundle latency | 198.52s |
| Maximum bundle latency | 292.26s |
| Total evaluation duration | 3971.48s |
