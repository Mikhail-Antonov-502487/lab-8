import cv2
import numpy as np

def find_contour_center(frame, lower_color, upper_color):
    """Поиск центра контура по цвету"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Берем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY, largest_contour
    
    return None, None, None

def main():
    marker_img = cv2.imread('variant-7.jpg')
    
    if marker_img is None:
        print("Ошибка: не удалось загрузить изображение метки")
        return
    
    # Отражаем по горизонтали и переворачиваем вертикально
    marker_transformed = cv2.flip(marker_img, -1)
    cv2.imwrite('marker_transformed.jpg', marker_transformed)
    print("Изображение преобразовано (отражение по обеим осям)")
    
    # Захват с камеры
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    # Поиск желтой метки:
    lower_color = np.array([20, 100, 100])
    upper_color = np.array([30, 255, 255])
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Рисуем центр кадра
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.line(frame, (center_x - 20, center_y), 
                 (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), 
                 (center_x, center_y + 20), (0, 0, 255), 2)
        
        # Поиск метки
        cX, cY, contour = find_contour_center(frame, lower_color,
                                               upper_color)
        
        if cX is not None and cY is not None:
            # Рисуем контур
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Рисуем центр метки
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            
            # Вычисляем расстояние до центра
            distance = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
            
            cv2.putText(frame, f"Distance: {distance:.1f} px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Marker: ({cX}, {cY})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Рисуем линию
            cv2.line(frame, (cX, cY), (center_x, center_y),
                      (255, 0, 255), 2)
        
        cv2.imshow('Tracking - Distance to Center', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()