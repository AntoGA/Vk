# Inference integration — 25 октября 2026

База: `6132d1cd77d5c6247cab83df19cc3bc851a03cde`, PR #6,
`fix/pr5-reproducible-contract`. Продолжение core audit; этот отчёт обновляет
его ограничения в отношении CPU BatchProcessor и ModelOptimizer.

## Воспроизведённые ошибки до правки

С исходными модулями и текущей небольшой моделью:

- `BatchProcessor._process_batch`: `AttributeError: 'list' object has no attribute 'to'`.
  В service передавался список, хотя требовался ModelInput; ModelOutput также
  ошибочно обрабатывался как список результатов.
- `ModelOptimizer.optimize_for_inference`: `TypeError: InterestPredictor.forward()
  got an unexpected keyword argument 'profile_features'`.
  Warmup и benchmark использовали старый контракт и жёстко заданные размерности.
- Worker ловил любые исключения получения очереди, проглатывал ошибки обработки,
  не передавая их ожидающему клиенту; повторный start мог запускать второго worker.
- Остановка не гарантировала обработку уже принятых запросов.
- Докстроки обещали <100ms без измеренного доказательства.

## Что изменено

- BatchRequest.features теперь ModelInput с batch_size=1.
- Collation дополняет action sequence и boolean mask до общей длины;
  графы должны иметь одинаковое число узлов. Без graph-node mask разный размер
  графов явно отклоняется, чтобы padding не менял предсказания незаметно.
- ModelOutput разбивается на отдельные single-item результаты.
- Очередь использует monotonic deadline, Condition для результатов и sentinel
  для остановки; повторный start идемпотентен, duplicate request_id отклоняется
  до получения предыдущего результата.
- Ошибки batch доступны через get_result. Ошибка callback логируется и не
  останавливает worker. stop прекращает приём, обрабатывает принятую очередь,
  затем завершает worker; после 5 секунд ожидания явно возвращает TimeoutError.
- Optimizer создаёт пример из model.config или принимает example_input;
  использует model(ModelInput), eval/inference_mode и синхронизацию CUDA вокруг
  измерения. Подготовка больше не отключает requires_grad навсегда.
- optimized=True выставляется только после успешного warmup. Ошибки compile
  не замалчиваются. FP16 на CPU отклоняется до преобразования весов.
- Непроверенный ONNX путь явно отключён через NotImplementedError; это не
  реализация или подтверждение работоспособности экспорта.

## Проверки

Локально: Python 3.12.14, torch 2.14.1+cpu, OMP/MKL_NUM_THREADS=1.
`PYTHONPATH=. python -m pytest -q tests/ --junitxml=artifacts/tests.xml`:
**25 passed, 6.83 s**. `python -m compileall -q ml scripts tests`: успешно.

10 добавленных проверок: эквивалентность batch и одиночного inference с разной
длиной sequence и draining при stop; передача ошибки модели и restart; duplicate
и submission до start; изоляция callback failure; отклонение разных graph sizes;
optimizer preparation и finite benchmark; пример из конфигурации; два случая
невалидного n_runs; явные ошибки CPU FP16 и ONNX.

CI запускает весь tests/, E2E и checkpoint benchmark на Python 3.11/3.12.
JUnit и E2E/benchmark JSON сохраняются в `poc-evidence-python-*` в Actions.
Успех CI подтверждается отдельно по итоговому запуску, а не подразумевается этим текстом.

## Ограничения

- API/dashboard/parquet data module и legacy train/evaluate scripts ещё требуют
  отдельной адаптации и integration tests.
- ONNX export не реализован; CUDA/FP16/torch.compile не проверены в CPU CI.
- Нет graph-node padding mask, разделения несовместимых запросов по batch-группам,
  backpressure и ограничений памяти очереди/неполученных результатов.
- Ошибка одного несовместимого input отклоняет весь собранный batch.
- Caller должен сохранять входные тензоры неизменными до обработки.
- Это проверка PoC; real VK data, нагрузка и SLA не проверялись.

`main` не изменяется, merge не выполняется, PR остаётся draft.
