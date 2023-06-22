import cv2
import numpy as np
import onnxruntime

# Создание сессии ONNX Runtime и загрузка модели
session = onnxruntime.InferenceSession("C:/Users/lolol/OneDrive/Документы/Курсовой проект/Diploma project (Дилом)/src/CNN model/person_detection_model.onnx")

# Получение имен входных и выходных тензоров модели
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Открытие видеопотока с камеры
cap = cv2.VideoCapture(0)

while True:
    # Считывание кадра из видеопотока
    ret, frame = cap.read()
    
    # Если кадр успешно считан
    if ret:
        # Изменение размера кадра до 32x32 пикселя
        frame_resized = cv2.resize(frame, (32, 32))
        
        # Преобразование кадра в формат RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Преобразование кадра в формат float32 и нормализация значений пикселей
        frame_float = frame_rgb.astype(np.float32) / 255.0
        
        # Создание входного тензора ONNX
        input_tensor = frame_float.transpose(2, 0, 1)  # Изменение порядка осей
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Добавление оси пакета
        
        # Выполнение предсказания модели
        output = session.run([output_name], {input_name: input_tensor})
        
        # Извлечение класса с наивысшей вероятностью
        class_index = np.argmax(output)
        print(class_index)
        # Отображение результата на кадре
        if class_index == 1:
            cv2.putText(frame, "Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Отображение кадра
        cv2.imshow("Frame", frame)
    
    # Прерывание выполнения по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
