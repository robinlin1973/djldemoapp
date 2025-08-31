package com.example.djldemoapp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;

import java.util.List;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.IntStream;

public class TestTorchModel {

    /**
     * Usage:
     *   java ... TestTorchModel /path/to/model.pt [cpu|gpu]
     *
     * Notes:
     * - Model must be a TorchScript file saved via torch.jit.trace/script.
     * - Input shape below is a demo (1,3,224,224). Adjust to your model.
     */
    public static void main(String[] args) throws Exception {
    	ClassLoader loader = Thread.currentThread().getContextClassLoader();

    	Path modelPath = Paths.get(loader.getResource("model/resnet18.pt").toURI());
//        Path modelPath = Paths.get("/tmp/resnet18.pt");
        
        boolean useGpu = args.length > 1 && args[1].equalsIgnoreCase("gpu");

        // Pick device
        Device device = useGpu ? Device.gpu() : Device.cpu();
        System.out.println("[INFO] Engine: PyTorch, Device: " + device);

        // Minimal pass-through translator: NDList -> NDList
        Translator<NDList, NDList> passthrough = new Translator<>() {
    	  @Override public NDList processInput(TranslatorContext ctx, NDList in) { return in; }
    	  @Override public NDList processOutput(TranslatorContext ctx, NDList out) { return out; }
    	  @Override public Batchifier getBatchifier() { return null; } // no [1, ...] added
    	};

        // Build criteria to load a local TorchScript file
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(modelPath.getParent())
                .optModelName(modelPath.getFileName().toString()) // e.g., "model.pt"
                .optEngine("PyTorch")
                .optDevice(device)
                .optTranslator(passthrough)
                .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
             Predictor<NDList, NDList> predictor = model.newPredictor();
             NDManager manager = NDManager.newBaseManager(device)) {

        	// --- DEMO INPUT (adjust to your real model) ---
        	Path imgPath = Paths.get(loader.getResource("img/kitten.jpg").toURI());
        	Image img = ImageFactory.getInstance().fromFile(imgPath);

        	NDArray x = img.toNDArray(manager);                    // HWC, uint8
        	x = NDImageUtils.resize(x, 224, 224);                  // HWC
        	x = NDImageUtils.toTensor(x);                          // CHW, float32 in [0,1]
        	x = NDImageUtils.normalize(
        	        x,
        	        new float[]{0.485f, 0.456f, 0.406f},           // mean
        	        new float[]{0.229f, 0.224f, 0.225f});          // std
        	x = x.expandDims(0);                                   // NCHW -> [1,3,224,224]
            NDList input = new NDList(x);
            NDList out = predictor.predict(input);
            
            NDArray logits = out.head();                 // shape (1, 1000)
            NDArray probsNd = logits.softmax(1).flatten();      // shape (1000)
            float[] probs = probsNd.toFloatArray();

            // get top-5 indices by probability
            int[] top5 = IntStream.range(0, probs.length)
                    .boxed()
                    .sorted((a,b) -> Float.compare(probs[b], probs[a]))
                    .limit(5)
                    .mapToInt(Integer::intValue)
                    .toArray();

            Path labelPath = Paths.get(loader.getResource("label/synset.txt").toURI());
            List<String> labels = Files.readAllLines(labelPath); // 1000 ImageNet labels
            for (int idx : top5) {
                System.out.printf("  %d : %-25s %.4f%n", idx, labels.get(idx), probs[idx]);
            }
        } catch (UnsatisfiedLinkError ule) {
            System.err.println("Native library load failed. Check you included the correct DJL PyTorch native:\n" +
                    "- CPU: ai.djl.pytorch:pytorch-native-cpu:2.4.0 (classifier matches OS/arch)\n" +
                    "- CUDA: ai.djl.pytorch:pytorch-native-cuXXX:2.4.0 (classifier matches OS/arch)\n" +
                    "Also confirm libc/glibc and CUDA drivers match the native build.");
            throw ule;
        } catch (Exception e) {
            System.err.println("Failed to run inference. Common causes:\n" +
                    "- The .pt is not TorchScript (export with torch.jit.script/trace)\n" +
                    "- Input tensor shape/type doesn’t match the model\n" +
                    "- Wrong device or missing CUDA libraries\n" +
                    "- Model filename/path mismatch");
            throw e;
        }
    }
}
