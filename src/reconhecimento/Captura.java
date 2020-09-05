package reconhecimento;

import java.awt.event.KeyEvent;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class Captura {
	
	public static void main(String[] args) throws Exception {
		
		KeyEvent tecla = null;
		
		OpenCVFrameConverter.ToMat converterMat  = new OpenCVFrameConverter.ToMat();
		
		OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
		
		camera.start();
		
		CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
		
		Frame frameCapturado = null;
		
		while((frameCapturado = camera.grab()) != null) {
			if(cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}
		}
		
		cFrame.dispose();
		camera.stop();
	}

}
