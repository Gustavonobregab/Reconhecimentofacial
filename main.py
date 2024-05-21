import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
tempodistraiu = 0
distraido = False

# Pegando webcam e conferindo a abertura dela 
webcam = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Não estou lendo a webcam")
            continue

        # Convertendo as cores porque o mediapipe pede
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        coordenadas = results.multi_face_landmarks

        
        if coordenadas:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

       
        image = cv2.flip(image, 1)

        if coordenadas:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])

        if (x < 310 or x > 410) or (y < 180 or y > 230):
            if not distraido:
                tempodistraiu += 1
                distraido = True
            cv2.putText(image, 'Distraido', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            distraido = False
            cv2.putText(image, 'Atento', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'Contador: {tempodistraiu}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'({x}, {y})', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


       

        # Exibindo a imagem com texto espelhado
        cv2.imshow('MediaPipe Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

webcam.release()
cv2.destroyAllWindows()
