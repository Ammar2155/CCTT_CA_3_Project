package org.cloudbus.cloudsim.examples;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.DoubleStream;

public class T_Test_Validation {

    private static final double OMEGA_VIRTUAL = 0.10;

    private static HPCClassifierLoader classifier;

    public static void main(String[] args) {
        Log.disable();
        System.out.println("Starting HPNTS Simulation with T-Test Validation...");

        try {

            classifier = HPCClassifierLoader.fromJsonFile("hpc_classifier.json");
            System.out.println("Loaded classifier, feature order: " + classifier.getFeatureOrder());

            int[] testScenarios = { 50, 100, 150, 200, 250 };
            double[] baselineResults = new double[testScenarios.length];
            double[] hpntsResults = new double[testScenarios.length];

            System.out.printf("%-15s | %-15s | %-15s\n", "Tasks", "Baseline (s)", "HPNTS (s)");
            System.out.println("-------------------------------------------------------------");

            for (int i = 0; i < testScenarios.length; i++) {
                int taskCount = testScenarios[i];

                baselineResults[i] = runSimulation(false, taskCount);
                hpntsResults[i] = runSimulation(true, taskCount);

                System.out.printf("%-15d | %-15.2f | %-15.2f\n", taskCount, baselineResults[i], hpntsResults[i]);
            }

            performStatisticalValidation(baselineResults, hpntsResults);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double runSimulation(boolean isHPNTS, int taskCount) throws Exception {
        CloudSim.init(1, Calendar.getInstance(), false);
        Datacenter datacenter = createDatacenter("Datacenter_0");
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        List<Vm> vmlist = new ArrayList<>();
        List<Vm> flatVms = new ArrayList<>();
        List<Vm> nestedContainers = new ArrayList<>();

        for (int i = 0; i < 2; i++) {
            Vm flatVm = new Vm(i, brokerId, 1000, 2, 2048, 1000, 10000, "Xen", new CloudletSchedulerTimeShared());
            flatVms.add(flatVm);
            vmlist.add(flatVm);
        }
        for (int i = 2; i < 10; i++) {
            Vm container = new Vm(i, brokerId, 250, 1, 512, 500, 5000, "Docker", new CloudletSchedulerTimeShared());
            nestedContainers.add(container);
            vmlist.add(container);
        }

        broker.submitGuestList(vmlist);
        List<Cloudlet> cloudletList = new ArrayList<>();
        long baseLength = 15000;
        UtilizationModel utilizationModel = new UtilizationModelFull();
        Random featureRng = new Random(1234);

        for (int id = 0; id < taskCount; id++) {
            double cpuRequest = featureRng.nextDouble();
            double memRequest = featureRng.nextDouble();
            double durationNorm = featureRng.nextDouble();
            double ioIntensity = featureRng.nextDouble();
            double[] features = { cpuRequest, memRequest, durationNorm, ioIntensity };
            boolean isHPCTask = classifier.isHPC(features);

            Cloudlet task;
            if (isHPCTask) {
                task = new Cloudlet(id, baseLength, 2, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                task.setClassType(1);
            } else {
                long length = isHPNTS ? (long) (baseLength * (1 + OMEGA_VIRTUAL)) : baseLength;
                task = new Cloudlet(id, length, 1, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                task.setClassType(0);
            }
            cloudletList.add(task);
        }

        if (isHPNTS) {
            cloudletList.sort((c1, c2) -> Integer.compare(c2.getClassType(), c1.getClassType()));
        }

        broker.submitCloudletList(cloudletList);

        if (isHPNTS) {
            int flatIndex = 0;
            int containerIndex = 0;
            for (Cloudlet task : cloudletList) {
                if (task.getClassType() == 1) {
                    broker.bindCloudletToVm(task.getCloudletId(), flatVms.get(flatIndex).getId());
                    flatIndex = (flatIndex + 1) % flatVms.size();
                } else {
                    broker.bindCloudletToVm(task.getCloudletId(), nestedContainers.get(containerIndex).getId());
                    containerIndex = (containerIndex + 1) % nestedContainers.size();
                }
            }
        }

        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        List<Cloudlet> finishedList = broker.getCloudletReceivedList();
        double makespan = 0;
        for (Cloudlet cl : finishedList) {
            if (cl.getFinishTime() > makespan)
                makespan = cl.getFinishTime();
        }
        return makespan;
    }

    private static void performStatisticalValidation(double[] groupA, double[] groupB) {
        int n = groupA.length;
        double[] diff = new double[n];
        for (int i = 0; i < n; i++) {
            diff[i] = groupA[i] - groupB[i];
        }

        double meanDiff = DoubleStream.of(diff).sum() / n;
        double sumSqDiff = 0;
        for (double d : diff) {
            sumSqDiff += Math.pow(d - meanDiff, 2);
        }

        double stdDev = Math.sqrt(sumSqDiff / (n - 1));
        double standardError = stdDev / Math.sqrt(n);
        double tStat = meanDiff / standardError;

        System.out.println("\n=======================================================");
        System.out.println("   STATISTICAL VALIDATION: PAIRED T-TEST RESULTS");
        System.out.println("=======================================================");
        System.out.printf("Mean Difference: %.4f\n", meanDiff);
        System.out.printf("Standard Deviation: %.4f\n", stdDev);
        System.out.printf("T-Statistic: %.4f\n", tStat);
        System.out.println("Degrees of Freedom: " + (n - 1));

        System.out.println("Critical T-Value (p<0.05, df=4): 2.776");

        if (Math.abs(tStat) > 2.776) {
            System.out.println("CONCLUSION: Statistically Significant (Reject Null Hypothesis)");
        } else {
            System.out.println("CONCLUSION: Not Statistically Significant (Fail to Reject Null)");
        }
        System.out.println("=======================================================");
    }

    private static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < 8; p++) {
                peList.add(new Pe(p, new PeProvisionerSimple(5000)));
            }
            hostList.add(new Host(i, new RamProvisionerSimple(32000), new BwProvisionerSimple(1000000), 100000, peList,
                    new VmSchedulerTimeShared(peList)));
        }
        return new Datacenter(name,
                new DatacenterCharacteristics("x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.001, 0.0),
                new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }
}
