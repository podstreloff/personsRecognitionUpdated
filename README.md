Email: podstrelov99@gmail.com
Telegram: @npodstrelov

Создать виртуальное окружение:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

Установить зависимости:
   python -m pip install --upgrade pip
   pip install -r requirements.txt

Скопировать файл crowd.mp4 в корень проекта:
crowd_detection_repo/crowd.mp4

YOLO (модель 1):
python -m src.infer --model yolo --input crowd.mp4 --output out_yolo.mp4 --device auto --conf 0.30 --resize 1280 --half

Faster R-CNN (модель 2):
python -m src.infer --model detr --input crowd.mp4 --output out_detr.mp4 --device auto --conf 0.30 --resize 1280


запуск с деноизером:

python -m src.infer --model yolo --input crowd.mp4 --output out_yolo_balanced.mp4 --device auto --conf 0.30 --resize 960 
--half --denoise fastnlm --denoise-nlm-h 6 --denoise-nlm-h-color 6

python -m src.infer --model detr --input crowd.mp4 --output out_detr_balanced.mp4 --device auto --conf 0.30 --resize 960 
--denoise fastnlm --denoise-nlm-h 6 --denoise-nlm-h-color 6 --denoise-nlm-template 7 --denoise-nlm-search 21

Оборудование: NVIDIA RTX 2060 (CUDA), AMD Ryzen 7 4800H; Python 3.10; PyTorch 2.1; Torchvision 0.16; Ultralytics 8.x.
Порог детекции: conf = 0.30 (можно изменить)
YOLO- одноэтапная архитектура (single-stage, anchor-free).
Faster R-CNN- двухэтапная (region proposals + head).


YOLO
по количеству найденных людей модели сопоставимы; у Faster R-CNN немного аккуратнее боксы на крупных объектах, но YOLOв8n заметно стабильнее и значительно быстрее

YOLOv8n

Плюсы: высокая скорость (real-time), плавные боксы (высокий temporal IoU), уверенная работа на средней/дальней дистанции, хорошо переносится на FP16/ONNX/TensorRT.
Минусы: в очень плотных перекрытиях иногда «слипание» близких людей; изредка ложные срабатывания на постеры/тени при низком conf (который можно изменить).

Faster R-CNN

Плюсы: классическая двухэтапная схема, точная локализация на крупных/контрастных фигурах.
Минусы: заметно ниже FPS, выше латентность, больше «мерцаний» на быстрых движениях/панорамировании.

Шумоподавление:
fastnlm(кадровый) заметно улучшает качество видео без сильного ущерба для FPS (YOLO ~8–12 FPS при --resize 960), улучшает стабильность;
fastnlm_multi (темпоральный, окно 3–5) даёт наиболее гладкую картинку и повышает temporal IoU, но снижает скорость сильнее.

Выбор предпочтительного алгоритма
Рекомендация: YOLOv8n — как основной детектор людей для видеопотока.
Обоснование:
лучшее соотношение качество/скорость (≈19 FPS при 1280 и FP16 на RTX 2060);
высокая темпоральная стабильность (IoU ≈ 0.925 на реальном ролике);
простая интеграция и масштабирование (FP16, ONNX/TensorRT, INT8-квантование);
удобная настройка по порогу/размеру входа

Faster R-CNN логично оставить как baseline или офлайновый «референс» для точностной проверки (например, в отчётах, где скорость не критична).

Шаги по улучшению качества и производительности
Тонкая настройка инференса: --conf (0.25–0.4), --resize (960–1280), NMS-параметры.
Шумоподавление:
быстрый вариант — --denoise bilateral (почти без потерь по FPS);
качественный — --denoise fastnlm (баланс), или --denoise fastnlm_multi --denoise-multi-window 3–5 (лучшее сглаживание во времени).
Трекинг поверх детекции (ByteTrack / OC-SORT / DeepSORT) — уменьшает мерцание, даёт ID-устойчивость, помогает с пропусками.
Сегментация (YOLOv8-seg/YOLOv9-seg) — точнее при сильных взаимных перекрытиях.
Доменно-специфичное дообучение (CrowdHuman, CityPersons, MOT17) — даже 1–3 эпохи часто ощутимо помогают.

Производительность
Полупряники/инференс-бэкенды: экспорт в ONNX/TensorRT/OpenVINO (ускорение ×1.5–3).
Снижение нагрузки: --resize 960, FP16, при необходимости INT8-квантование.
Конвейеризация: разнести I/O, Denoise и Inference по потокам; при офлайне — пакетная обработка нескольких кадров.
Адаптивные профили: на статике включать fastnlm_multi, на быстрых движениях — переключаться на bilateral/fastnlm.
