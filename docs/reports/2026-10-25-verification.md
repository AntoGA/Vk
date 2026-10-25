# Итоговая проверка PoC — 25 октября 2026

Проверен commit `65bd79d79a9699251ec3cb784c0be1fbaaaa5dbd` PR #6.
Содержимое локальных файлов сверено по Git blob SHA с деревом этого commit:
расхождений среди загруженных файлов нет. Локально загружены ML/scripts/tests,
README и конфигурация CI; это не полный checkout API/dashboard/deployment.

## Результаты

| Проверка | Результат |
| --- | --- |
| Полный tests/ | 25 passed, 6.86 s |
| Compileall ml/scripts/tests | Успешно |
| Synthetic E2E, CPU, 32 users, 1 epoch, batch 8 | Два успешных запуска |
| Повторяемость при seed 42 | Конфигурации, все тензоры checkpoint и quality metrics идентичны |
| Очередь: 4 параллельных отправителя, 64 запроса, batch 8 | 64 корректных результата; после stop нет pending/results, worker завершён |
| Checkpoint benchmark, warmup 3 / measured 10 | JSON создан; latency конечны и >= 0, throughput конечен и > 0 |
| CI Python 3.11 / 3.12 на проверенном commit | Success, тесты + E2E + benchmark + artifacts |

[Проверенный CI](https://github.com/AntoGA/Vk/actions/runs/40210307861).
JUnit и JSON доступны в artifacts `poc-evidence-python-3.11` и
`poc-evidence-python-3.12`. Структурированная сводка:
[2026-10-25-verification.json](2026-10-25-verification.json).

## Воспроизведение локальных проверок

Из корня репозитория с CPU torch, numpy и pytest:

```bash
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m compileall -q ml scripts tests
python -m pytest -q tests/ --junitxml=artifacts/tests.xml
python scripts/run_e2e_pipeline.py --num-users 32 --epochs 1 --batch-size 8 --device cpu --output-dir artifacts/verify-e2e
python scripts/run_e2e_pipeline.py --num-users 32 --epochs 1 --batch-size 8 --device cpu --output-dir artifacts/verify-e2e-repeat
python scripts/benchmark.py --model artifacts/verify-e2e/model.pt --device cpu --warmup 3 --num-runs 10 --output artifacts/verify-benchmark.json
```

При сравнении E2E latency и пути checkpoint исключаются: время закономерно
различается. Сравниваются quality, model_config и каждый tensor state_dict через
torch.equal. Byte-level равенство файлов checkpoint не заявляется.

## Вывод и границы

В выполненных проверках новых сбоев не обнаружено. Изменений исполняемого кода
этот этап не вносит; добавлены только отчёт и структурированная сводка.
Локальный snapshot без .git возвращает benchmark commit_sha=null — это ожидаемое
ограничение окружения, поэтому локальные значения скорости не выдаются за
commit-bound benchmark. CI содержит SHA и остаётся основным сохранённым доказательством.

Проверки не распространяются на реальный VK dataset, CUDA/FP16/torch.compile,
ONNX, API/dashboard, production deployment и нагрузочный SLA. 64 запроса —
функциональная проверка очереди, а не нагрузочная сертификация. Прочие известные
ограничения перечислены в core и inference integration reports.

`main` не изменён. PR #6 остаётся draft; merge не выполнялся.
