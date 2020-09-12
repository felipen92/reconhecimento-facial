package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;



import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_face.*;

public class Reconhecimento {

	private static OpenCVFrameGrabber camera;
	private static CascadeClassifier detectorFaces;
	private static Mat imagemColorida;

	public static void main(String[] args) throws Exception, InterruptedException {

		OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

		camera = new OpenCVFrameGrabber(0);

		String [] pessoas = {"", "Felipe", "Joao"};
		
		camera.start();

		detectorFaces = new CascadeClassifier("src\\recursos\\haarcascade_frontalface_alt.xml");
		
		//RECONHECIMENTO FRACO
        //FaceRecognizer reconhecedor = EigenFaceRecognizer.create();             
        //reconhecedor.read("src\\recursos\\classificadorEigenFaces.yml");        
        
        //SETAR A PRECISAO DO RECONHECIMENTO
        //reconhecedor.setThreshold(0);
        
        //RECONHECIMENTO MEDIO
        //FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
        //reconhecedor.read("src\\recursos\\classificadorFisherFaces.yml");
        
        //RECONHECIMENTO FORTE
        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        reconhecedor.read("src\\recursos\\classificadorLBPH.yml");

		CanvasFrame cFrame = new CanvasFrame("Reconhecimento", CanvasFrame.getDefaultGamma() / camera.getGamma());

		Frame frameCapturado = null;

		imagemColorida = new Mat();

		while ((frameCapturado = camera.grab()) != null) {

			imagemColorida = converteMat.convert(frameCapturado);

			Mat imagemCinza = new Mat();

			cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);

			RectVector facesDesctadas = new RectVector();

			detectorFaces.detectMultiScale(imagemCinza, facesDesctadas, 1.1, 1, 0, new Size(150, 150),
					new Size(500, 500));

			for (int i = 0; i < facesDesctadas.size(); i++) {
				Rect dadosFaces = facesDesctadas.get(0);
				rectangle(imagemColorida, dadosFaces, new Scalar(0, 0, 255, 0));
				Mat faceCapturada = new Mat(imagemCinza, dadosFaces);
				resize(faceCapturada, faceCapturada, new Size(160, 160));
				
				IntPointer rotulo = new IntPointer(1);
				DoublePointer confianca = new DoublePointer(1);
				reconhecedor.predict(faceCapturada, rotulo, confianca);
				int predicao = rotulo.get(0);
				
				String nome;
				
				if(predicao == -1) {
					nome = "Desconhecido";
				}else {
					nome = pessoas[predicao] + " - " + confianca.get(0);
				}
				
				int x = Math.max(dadosFaces.tl().x(), 0);
				int y = Math.max(dadosFaces.tl().y(), 0);
				putText(imagemColorida, nome, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0, 255, 0, 0));

			}

			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}

		}

		cFrame.dispose();
		camera.stop();
	}

}
