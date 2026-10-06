package org.cloudbus.cloudsim.examples;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

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

    public boolean isHPC(double[] features) {
        double z = bias;
        for (int i = 0; i < weights.length; i++) z += weights[i] * features[i];
        double p = 1.0 / (1.0 + Math.exp(-z));
        return p >= threshold;
    }

    public List<String> getFeatureOrder() {
        return featureOrder;
    }

    public static void exampleUsage() throws IOException {
        HPCClassifierLoader classifier = HPCClassifierLoader.fromJsonFile("hpc_classifier.json");
        double cpuRequest = 0.62, memRequest = 0.41, durationNorm = 0.30, ioIntensity = 0.15;
        double[] features = {cpuRequest, memRequest, durationNorm, ioIntensity};
        boolean isHPC = classifier.isHPC(features);
        System.out.println("Classified as: " + (isHPC ? "HPC" : "HTC"));
    }
}
