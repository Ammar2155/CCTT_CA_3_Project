package org.cloudbus.cloudsim.examples;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * HPCClassifierLoader
 *
 * Loads the JSON coefficients exported by classifier/train_classifier.py and
 * applies logistic-regression inference to classify a Cloudlet as HPC (1) or
 * HTC (0). Drop-in replacement for:
 *
 *     if (id % 5 == 0) { ... HPC ... } else { ... HTC ... }
 *
 * in both T_Test_Validation.java and HPNTS_Comparative_Project.java.
 *
 * No external JSON library dependency — the exported file is small and fixed
 * shape, so a minimal regex-based parse is enough. Swap for org.json/Jackson
 * if either is already on your classpath.
 */
public class HPCClassifierLoader {

    private final double[] weights;
    private final double bias;
    private final double threshold;
    private final List<String> featureOrder;

    private HPCClassifierLoader(double[] weights, double bias, double threshold, List<String> featureOrder) {
        this.weights = weights;
        this.bias = bias;
        this.threshold = threshold;
        this.featureOrder = featureOrder;
    }

    public static HPCClassifierLoader fromJsonFile(String path) throws IOException {
        Path p = Paths.get(path);
        if (!Files.exists(p)) {
            Path userDirP = Paths.get(System.getProperty("user.dir"), path);
            if (Files.exists(userDirP)) {
                p = userDirP;
            } else {
                Path fileNameP = Paths.get(System.getProperty("user.dir"), p.getFileName().toString());
                if (Files.exists(fileNameP)) {
                    p = fileNameP;
                }
            }
        }

        if (!Files.exists(p)) {
            throw new FileNotFoundException("Could not locate classifier JSON file at '" + path + 
                "' or relative to working directory '" + System.getProperty("user.dir") + "'");
        }

        String content = new String(Files.readAllBytes(p));

        List<String> featureOrder = new ArrayList<>();
        Matcher fm = Pattern.compile("\"feature_order\":\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(content);
        if (fm.find()) {
            for (String s : fm.group(1).split(",")) {
                featureOrder.add(s.trim().replaceAll("\"", ""));
            }
        }

        Matcher wm = Pattern.compile("\"weights\":\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(content);
        double[] weights = new double[0];
        if (wm.find()) {
            String[] parts = wm.group(1).split(",");
            weights = new double[parts.length];
            for (int i = 0; i < parts.length; i++) weights[i] = Double.parseDouble(parts[i].trim());
        }

        Matcher bm = Pattern.compile("\"bias\":\\s*([-0-9.eE]+)").matcher(content);
        double bias = bm.find() ? Double.parseDouble(bm.group(1)) : 0.0;

        Matcher tm = Pattern.compile("\"threshold\":\\s*([-0-9.eE]+)").matcher(content);
        double threshold = tm.find() ? Double.parseDouble(tm.group(1)) : 0.5;

        return new HPCClassifierLoader(weights, bias, threshold, featureOrder);
    }

    /** features must be supplied in the SAME order as featureOrder (see getFeatureOrder()). */
    public boolean isHPC(double[] features) {
        double z = bias;
        for (int i = 0; i < weights.length; i++) z += weights[i] * features[i];
        double p = 1.0 / (1.0 + Math.exp(-z));
        return p >= threshold;
    }

    public List<String> getFeatureOrder() {
        return featureOrder;
    }

    /**
     * Example integration inside runSimulation(), replacing:
     *     if (id % 5 == 0) { task.setClassType(1); } else { task.setClassType(0); }
     *
     * You'll need real per-task cpu_request / mem_request / duration_norm /
     * io_intensity values here -- derive them from your Cloudlet generation
     * parameters (or from replayed trace rows) rather than placeholders.
     */
    public static void exampleUsage() throws IOException {
        HPCClassifierLoader classifier = HPCClassifierLoader.fromJsonFile("hpc_classifier.json");
        double cpuRequest = 0.62, memRequest = 0.41, durationNorm = 0.30, ioIntensity = 0.15;
        double[] features = {cpuRequest, memRequest, durationNorm, ioIntensity}; // must match feature_order
        boolean isHPC = classifier.isHPC(features);
        System.out.println("Classified as: " + (isHPC ? "HPC" : "HTC"));
    }
}
