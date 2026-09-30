package org.cloudbus.cloudsim.examples;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import java.text.DecimalFormat;
import java.util.*;

/**
 * HPNTS Comparative Simulation for CloudSim 7.0.1 with DRL-TANS Engine Integration
 */
public class HPNTS_Comparative_Project {

    private static final int TASKS = 50;
    private static final double OMEGA_VIRTUAL = 0.10; 

    public static final int MODE_DETERMINISTIC_HPNTS = 0;
    public static final int MODE_DRL_TANS = 1;

    private static HPCClassifierLoader classifier;

    public static void main(String[] args) {
        Log.disable();

        boolean isTrain = false;
        boolean isEval = false;
        for (String arg : args) {
            if (arg.equals("--train")) isTrain = true;
            if (arg.equals("--eval")) isEval = true;
        }

        if (!isTrain && !isEval) {
            isTrain = true;
        }

        try {
            classifier = HPCClassifierLoader.fromJsonFile("hpc_classifier.json");

            double[] omegaVirtualArray = new double[]{0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10};

            if (isTrain) {
                System.out.println("Starting HPNTS CloudSim Training Bridge (300 Episodes)...");
                try (TANSBridgeClient bridge = new TANSBridgeClient("localhost", 8765, 10, 8, omegaVirtualArray)) {
                    System.out.println("[Java Bridge] Connected to TANS server on localhost:8765");
                    int trainEpisodes = 300;
                    for (int ep = 0; ep < trainEpisodes; ep++) {
                        int episodeSeed = 1000 + ep; // DIFFERENT seed per episode for training
                        double totalEpReward = runEpisode(MODE_DRL_TANS, true, bridge, episodeSeed, ep == 0);
                        if ((ep + 1) % 50 == 0 || ep == 0) {
                            System.out.printf("[Java Train] Ep %3d/%d finished | Seed=%d | Ep Total Step Reward=%.4f\n",
                                ep + 1, trainEpisodes, episodeSeed, totalEpReward);
                        }
                    }
                }
            } else if (isEval) {
                System.out.println("Starting Frozen-Policy Evaluation (10 Episodes per condition)...");
                try (TANSBridgeClient bridge = new TANSBridgeClient("localhost", 8765, 10, 8, omegaVirtualArray)) {
                    System.out.println("[Java Bridge] Connected to TANS server in EVAL mode on localhost:8765");

                    int evalEpisodes = 10;

                    // 1. Stationary Condition (seeds 5000..5009)
                    double[] hpntsStationary = new double[evalEpisodes];
                    double[] drlStationary = new double[evalEpisodes];

                    for (int ep = 0; ep < evalEpisodes; ep++) {
                        int seed = 5000 + ep;
                        hpntsStationary[ep] = runSimulationMakespan(MODE_DETERMINISTIC_HPNTS, false, null, seed);
                        drlStationary[ep] = runSimulationMakespan(MODE_DRL_TANS, false, bridge, seed);
                    }

                    // 2. Drifting Condition (seeds 6000..6009)
                    double[] hpntsDrifting = new double[evalEpisodes];
                    double[] drlDrifting = new double[evalEpisodes];

                    for (int ep = 0; ep < evalEpisodes; ep++) {
                        int seed = 6000 + ep;
                        hpntsDrifting[ep] = runSimulationMakespan(MODE_DETERMINISTIC_HPNTS, true, null, seed);
                        drlDrifting[ep] = runSimulationMakespan(MODE_DRL_TANS, true, bridge, seed);
                    }

                    printEvaluationTable(hpntsStationary, drlStationary, hpntsDrifting, drlDrifting);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double runEpisode(int mode, boolean isDrifting, TANSBridgeClient bridge, int episodeSeed, boolean printFirst5Rewards) throws Exception {
        CloudSim.init(1, Calendar.getInstance(), false);
        Datacenter datacenter = createDatacenter("Datacenter_0");
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        List<Vm> vmlist = createVmList(brokerId);
        broker.submitGuestList(vmlist);

        List<Cloudlet> cloudletList = generateCloudlets(brokerId, isDrifting, episodeSeed, mode != MODE_DETERMINISTIC_HPNTS);

        if (mode == MODE_DETERMINISTIC_HPNTS) {
            cloudletList.sort((c1, c2) -> Integer.compare(c2.getClassType(), c1.getClassType()));
        }

        broker.submitCloudletList(cloudletList);

        double totalEpisodeReward = 0.0;

        if (mode == MODE_DETERMINISTIC_HPNTS) {
            int flatIndex = 0;
            int containerIndex = 0;
            for (Cloudlet task : cloudletList) {
                if (task.getClassType() == 1) {
                    broker.bindCloudletToVm(task.getCloudletId(), vmlist.get(flatIndex % 2).getId());
                    flatIndex++;
                } else {
                    broker.bindCloudletToVm(task.getCloudletId(), vmlist.get(2 + (containerIndex % 8)).getId());
                    containerIndex++;
                }
            }
        } else if (mode == MODE_DRL_TANS && bridge != null) {
            double lastReward = 0.0;
            for (int i = 0; i < cloudletList.size(); i++) {
                Cloudlet task = cloudletList.get(i);

                double[] utilization = new double[10];
                double[] queueLength = new double[10];
                for (int r = 0; r < 10; r++) {
                    Vm vm = vmlist.get(r);
                    double util = vm.getTotalUtilizationOfCpu(CloudSim.clock());
                    utilization[r] = Math.min(1.0, Math.max(0.0, util));
                    int waitingCount = vm.getCloudletScheduler().getCloudletWaitingList().size();
                    queueLength[r] = Math.min(1.0, (double) waitingCount / 10.0);
                }
                bridge.updateHistory(utilization, queueLength);

                double priority = (double) task.getClassType();
                double miNorm = (double) task.getCloudletLength() / 30000.0;
                double queueDepthNorm = (double) i / (double) TASKS;
                double timeNorm = (double) i / (double) TASKS;
                double[] extraFeatures = new double[]{priority, miNorm, queueDepthNorm, timeNorm};

                boolean done = (i == cloudletList.size() - 1);

                if (printFirst5Rewards && i < 5) {
                    System.out.printf("  [Ep 1 Step %d Raw Reward] = %.4f (sign=%s, done=%b)\n",
                        i + 1, lastReward, (lastReward <= 0 ? "NEGATIVE" : "POSITIVE"), (i > 0 && i - 1 == cloudletList.size() - 1));
                }

                int chosenResourceIdx = bridge.chooseResource(extraFeatures, lastReward, done);
                chosenResourceIdx = Math.max(0, Math.min(9, chosenResourceIdx));

                Vm chosenVm = vmlist.get(chosenResourceIdx);
                broker.bindCloudletToVm(task.getCloudletId(), chosenVm.getId());

                double omega = (chosenResourceIdx >= 2) ? OMEGA_VIRTUAL : 0.0;
                double execTimeEst = (double) task.getCloudletLength() * (1.0 + omega) / (chosenVm.getMips() * chosenVm.getNumberOfPes());
                boolean isHpcMismatch = (task.getClassType() == 1 && chosenResourceIdx >= 2);
                lastReward = - (execTimeEst / 10.0) - (isHpcMismatch ? 0.5 : 0.0) - 0.1 * queueLength[chosenResourceIdx];
                totalEpisodeReward += lastReward;
            }
        }

        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        return totalEpisodeReward;
    }

    private static double runSimulationMakespan(int mode, boolean isDrifting, TANSBridgeClient bridge, int episodeSeed) throws Exception {
        CloudSim.init(1, Calendar.getInstance(), false);
        Datacenter datacenter = createDatacenter("Datacenter_0");
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        List<Vm> vmlist = createVmList(brokerId);
        broker.submitGuestList(vmlist);

        List<Cloudlet> cloudletList = generateCloudlets(brokerId, isDrifting, episodeSeed, mode != MODE_DETERMINISTIC_HPNTS);

        if (mode == MODE_DETERMINISTIC_HPNTS) {
            cloudletList.sort((c1, c2) -> Integer.compare(c2.getClassType(), c1.getClassType()));
        }

        broker.submitCloudletList(cloudletList);

        if (mode == MODE_DETERMINISTIC_HPNTS) {
            int flatIndex = 0;
            int containerIndex = 0;
            for (Cloudlet task : cloudletList) {
                if (task.getClassType() == 1) {
                    broker.bindCloudletToVm(task.getCloudletId(), vmlist.get(flatIndex % 2).getId());
                    flatIndex++;
                } else {
                    broker.bindCloudletToVm(task.getCloudletId(), vmlist.get(2 + (containerIndex % 8)).getId());
                    containerIndex++;
                }
            }
        } else if (mode == MODE_DRL_TANS && bridge != null) {
            double lastReward = 0.0;
            for (int i = 0; i < cloudletList.size(); i++) {
                Cloudlet task = cloudletList.get(i);

                double[] utilization = new double[10];
                double[] queueLength = new double[10];
                for (int r = 0; r < 10; r++) {
                    Vm vm = vmlist.get(r);
                    double util = vm.getTotalUtilizationOfCpu(CloudSim.clock());
                    utilization[r] = Math.min(1.0, Math.max(0.0, util));
                    int waitingCount = vm.getCloudletScheduler().getCloudletWaitingList().size();
                    queueLength[r] = Math.min(1.0, (double) waitingCount / 10.0);
                }
                bridge.updateHistory(utilization, queueLength);

                double priority = (double) task.getClassType();
                double miNorm = (double) task.getCloudletLength() / 30000.0;
                double queueDepthNorm = (double) i / (double) TASKS;
                double timeNorm = (double) i / (double) TASKS;
                double[] extraFeatures = new double[]{priority, miNorm, queueDepthNorm, timeNorm};

                boolean done = (i == cloudletList.size() - 1);
                int chosenResourceIdx = bridge.chooseResource(extraFeatures, lastReward, done);
                chosenResourceIdx = Math.max(0, Math.min(9, chosenResourceIdx));

                Vm chosenVm = vmlist.get(chosenResourceIdx);
                broker.bindCloudletToVm(task.getCloudletId(), chosenVm.getId());

                double omega = (chosenResourceIdx >= 2) ? OMEGA_VIRTUAL : 0.0;
                double execTimeEst = (double) task.getCloudletLength() * (1.0 + omega) / (chosenVm.getMips() * chosenVm.getNumberOfPes());
                boolean isHpcMismatch = (task.getClassType() == 1 && chosenResourceIdx >= 2);
                lastReward = - (execTimeEst / 10.0) - (isHpcMismatch ? 0.5 : 0.0) - 0.1 * queueLength[chosenResourceIdx];
            }
        }

        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        List<Cloudlet> finishedList = broker.getCloudletReceivedList();
        double makespan = 0;
        for (Cloudlet cl : finishedList) {
            if (cl.getFinishTime() > makespan) {
                makespan = cl.getFinishTime();
            }
        }
        return makespan;
    }

    private static List<Vm> createVmList(int brokerId) {
        List<Vm> vmlist = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            Vm flatVm = new Vm(i, brokerId, 1000, 2, 2048, 1000, 10000, "Xen", new CloudletSchedulerTimeShared());
            vmlist.add(flatVm);
        }
        for (int i = 2; i < 10; i++) {
            Vm container = new Vm(i, brokerId, 250, 1, 512, 500, 5000, "Docker", new CloudletSchedulerTimeShared());
            vmlist.add(container);
        }
        return vmlist;
    }

    private static List<Cloudlet> generateCloudlets(int brokerId, boolean isDrifting, int episodeSeed, boolean isDRL) {
        List<Cloudlet> cloudletList = new ArrayList<>();
        long baseLength = 15000;
        UtilizationModel utilizationModel = new UtilizationModelFull();
        Random rng = new Random(episodeSeed);

        for (int id = 0; id < TASKS; id++) {
            boolean isHPCTask;
            if (isDrifting) {
                double hpcProb;
                if (id < TASKS / 3) {
                    hpcProb = 0.20;
                } else if (id < (2 * TASKS) / 3) {
                    hpcProb = 0.60;
                } else {
                    hpcProb = 0.30;
                }
                isHPCTask = (rng.nextDouble() < hpcProb);
            } else {
                double cpuRequest = rng.nextDouble();
                double memRequest = rng.nextDouble();
                double durationNorm = rng.nextDouble();
                double ioIntensity = rng.nextDouble();
                double[] features = {cpuRequest, memRequest, durationNorm, ioIntensity};
                isHPCTask = classifier.isHPC(features);
            }

            Cloudlet task;
            if (isHPCTask) {
                task = new Cloudlet(id, baseLength, 2, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                task.setClassType(1);
            } else {
                long length = (long)(baseLength * (1 + OMEGA_VIRTUAL));
                task = new Cloudlet(id, length, 1, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                task.setClassType(0);
            }
            cloudletList.add(task);
        }
        return cloudletList;
    }

    private static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < 8; p++) {
                peList.add(new Pe(p, new PeProvisionerSimple(5000)));
            }
            hostList.add(new Host(i, new RamProvisionerSimple(32000), new BwProvisionerSimple(1000000), 100000, peList, new VmSchedulerTimeShared(peList)));
        }
        return new Datacenter(name, new DatacenterCharacteristics("x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.001, 0.0), new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }

    private static void printEvaluationTable(double[] hpntsStat, double[] drlStat, double[] hpntsDrift, double[] drlDrift) {
        double hpntsStatMean = mean(hpntsStat);
        double hpntsStatStd = std(hpntsStat);
        double drlStatMean = mean(drlStat);
        double drlStatStd = std(drlStat);
        double statImprov = ((hpntsStatMean - drlStatMean) / hpntsStatMean) * 100.0;

        double hpntsDriftMean = mean(hpntsDrift);
        double hpntsDriftStd = std(hpntsDrift);
        double drlDriftMean = mean(drlDrift);
        double drlDriftStd = std(drlDrift);
        double driftImprov = ((hpntsDriftMean - drlDriftMean) / hpntsDriftMean) * 100.0;

        DecimalFormat df = new DecimalFormat("0.00");

        System.out.println("\n=========================================================================================");
        System.out.println("   FROZEN-POLICY EVALUATION RESULTS (10 EVAL EPISODES PER CONDITION)");
        System.out.println("=========================================================================================");
        System.out.printf("%-20s | %-30s | %-30s | %-12s\n", "Workload Scenario", "Deterministic HPNTS (mean ± std)", "DRL-TANS (mean ± std)", "% Improvement");
        System.out.println("-----------------------------------------------------------------------------------------");
        System.out.printf("%-20s | %s ± %s                    | %s ± %s                    | %s%%\n",
            "Stationary", df.format(hpntsStatMean), df.format(hpntsStatStd), df.format(drlStatMean), df.format(drlStatStd), df.format(statImprov));
        System.out.printf("%-20s | %s ± %s                    | %s ± %s                    | %s%%\n",
            "Drifting", df.format(hpntsDriftMean), df.format(hpntsDriftStd), df.format(drlDriftMean), df.format(drlDriftStd), df.format(driftImprov));
        System.out.println("=========================================================================================");
    }

    private static double mean(double[] arr) {
        double sum = 0;
        for (double v : arr) sum += v;
        return sum / arr.length;
    }

    private static double std(double[] arr) {
        double m = mean(arr);
        double sumSq = 0;
        for (double v : arr) sumSq += Math.pow(v - m, 2);
        return Math.sqrt(sumSq / (arr.length - 1));
    }
}