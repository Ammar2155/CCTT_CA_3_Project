package org.cloudbus.cloudsim.examples;

import java.io.*;
import java.net.Socket;
import java.util.*;

/**
 * TANSBridgeClient
 *
 * Talks to the Python actor-critic TANS agent (tans_agent.py) over a plain
 * TCP socket, one JSON line per scheduling decision. Drop-in replacement for
 * the deterministic binding loop in HPNTS_Comparative_Project.runSimulation().
 *
 * Start the Python side first:
 *   python tans_agent.py --port 8765 --train        (training run)
 *   python tans_agent.py --port 8765 --eval ckpt.pt  (frozen-policy comparison run)
 *
 * This is a SKELETON: state encoding, reward computation, and the resource
 * index mapping must match whatever grid layout tans_agent.py expects. Keep
 * the two in lock-step as you iterate.
 */
public class TANSBridgeClient implements Closeable {

    private final Socket socket;
    private final BufferedReader in;
    private final BufferedWriter out;

    // Rolling history buffers for the state grid: [resource][channel][tick]
    private final int nResources;
    private final int window;
    private final Deque<double[]>[] utilHistory; // per-resource rolling utilization
    private final Deque<double[]>[] queueHistory; // per-resource rolling queue length
    private final double[] omegaVirtual; // per-resource virtualization overhead (0 for flat VMs)

    @SuppressWarnings("unchecked")
    public TANSBridgeClient(String host, int port, int nResources, int window, double[] omegaVirtual) throws IOException {
        this.socket = new Socket(host, port);
        this.in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        this.out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
        this.nResources = nResources;
        this.window = window;
        this.omegaVirtual = omegaVirtual;
        this.utilHistory = new Deque[nResources];
        this.queueHistory = new Deque[nResources];
        for (int i = 0; i < nResources; i++) {
            utilHistory[i] = new ArrayDeque<>();
            queueHistory[i] = new ArrayDeque<>();
            for (int t = 0; t < window; t++) {
                utilHistory[i].add(new double[]{0.0});
                queueHistory[i].add(new double[]{0.0});
            }
        }
    }

    /** Call once per tick with fresh utilization/queue readings before requesting an action. */
    public void updateHistory(double[] utilization, double[] queueLength) {
        for (int i = 0; i < nResources; i++) {
            utilHistory[i].addLast(new double[]{utilization[i]});
            utilHistory[i].removeFirst();
            queueHistory[i].addLast(new double[]{queueLength[i]});
            queueHistory[i].removeFirst();
        }
    }

    /**
     * Request a routing decision for the task at the head of the queue.
     * extraFeatures: [priority (0/1), miNormalized, queueDepthNormalized, timeNormalized]
     * reward: reward for the PREVIOUS decision (0 on the very first call of an episode)
     * done: true if this is the terminal decision of the episode (for training bookkeeping)
     */
    public int chooseResource(double[] extraFeatures, double reward, boolean done) throws IOException {
        double[][][] grid = buildGrid();

        StringBuilder sb = new StringBuilder();
        sb.append("{\"type\":\"step\",\"grid\":").append(toJsonGrid(grid));
        sb.append(",\"extra\":").append(toJsonArray(extraFeatures));
        sb.append(",\"reward\":").append(reward);
        sb.append(",\"done\":").append(done);
        sb.append("}\n");

        out.write(sb.toString());
        out.flush();

        String line = in.readLine();
        if (line == null) throw new IOException("TANS agent connection closed unexpectedly");
        // Minimal hand-rolled parse to avoid a JSON dependency; swap for a real
        // JSON library (org.json / Jackson) if one is already on your classpath.
        int actionIdx = line.indexOf("\"action\":");
        int start = actionIdx + "\"action\":".length();
        int end = line.indexOf(",", start);
        if (end < 0) end = line.indexOf("}", start);
        return Integer.parseInt(line.substring(start, end).trim());
    }

    public void requestCheckpoint(String path) throws IOException {
        out.write("{\"type\":\"checkpoint\",\"path\":\"" + path + "\"}\n");
        out.flush();
        in.readLine(); // ack
    }

    private double[][][] buildGrid() {
        // channels: [utilization, queueLength, omegaVirtual] x [nResources] x [window]
        double[][][] grid = new double[3][nResources][window];
        for (int r = 0; r < nResources; r++) {
            int t = 0;
            for (double[] u : utilHistory[r]) grid[0][r][t++] = u[0];
            t = 0;
            for (double[] q : queueHistory[r]) grid[1][r][t++] = q[0];
            Arrays.fill(grid[2][r], omegaVirtual[r]);
        }
        return grid;
    }

    private String toJsonGrid(double[][][] grid) {
        StringBuilder sb = new StringBuilder("[");
        for (int c = 0; c < grid.length; c++) {
            sb.append("[");
            for (int r = 0; r < grid[c].length; r++) {
                sb.append(toJsonArray(grid[c][r]));
                if (r < grid[c].length - 1) sb.append(",");
            }
            sb.append("]");
            if (c < grid.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    private String toJsonArray(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public void close() throws IOException {
        socket.close();
    }
}
