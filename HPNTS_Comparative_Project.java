package org.cloudbus.cloudsim.examples;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

/**
 * HPNTS Comparative Simulation for CloudSim 7.0.1 (7G Architecture)
 */
public class HPNTS_Comparative_Project {

    private static final int TASKS = 50;
    // HPNTS Virtualization Overhead Parameter (10% penalty for nested)
    private static final double OMEGA_VIRTUAL = 0.10; 

    public static void main(String[] args) {
        System.out.println("Starting HPNTS Project Simulation in CloudSim 7.0.1...");

        try {
            // ==========================================
            // SIMULATION A: BASELINE (FCFS)
            // ==========================================
            double baselineMakespan = runSimulation(false);

            // ==========================================
            // SIMULATION B: HPNTS (Priority + Smart Mapping)
            // ==========================================
            double hpntsMakespan = runSimulation(true);

            // Print Comparative Results
            printComparison(baselineMakespan, hpntsMakespan);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double runSimulation(boolean isHPNTS) throws Exception {
        // 1. Initialize CloudSim
        CloudSim.init(1, Calendar.getInstance(), false);

        // 2. Create Datacenter
        Datacenter datacenter = createDatacenter("Datacenter_0");

        // 3. Create Broker
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        // 4. Create VMs (Resources)
        List<Vm> vmlist = new ArrayList<>();
        List<Vm> flatVms = new ArrayList<>();
        List<Vm> nestedContainers = new ArrayList<>();

        // Create 2 "Flat" VMs (High Power, Low Overhead)
        for (int i = 0; i < 2; i++) {
            Vm flatVm = new Vm(i, brokerId, 1000, 2, 2048, 1000, 10000, "Xen", new CloudletSchedulerTimeShared());
            flatVms.add(flatVm);
            vmlist.add(flatVm);
        }

        // Create 8 "Nested Containers" (Lightweight, High Density)
        for (int i = 2; i < 10; i++) {
            Vm container = new Vm(i, brokerId, 250, 1, 512, 500, 5000, "Docker", new CloudletSchedulerTimeShared());
            nestedContainers.add(container);
            vmlist.add(container);
        }
        
        // ---> FIX APPLIED HERE: CloudSim 7G uses Guest architecture <---
        broker.submitGuestList(vmlist); 

        // 5. Create Cloudlets (Tasks)
        List<Cloudlet> cloudletList = new ArrayList<>();
        long baseLength = 15000;
        UtilizationModel utilizationModel = new UtilizationModelFull();

        for (int id = 0; id < TASKS; id++) {
            Cloudlet task;
            if (id >= 0) {
                // 20% of tasks are Critical HPC (Priority 1)
                task = new Cloudlet(id, baseLength, 2, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                // We use "class type" as a proxy for Priority in standard CloudSim
                task.setClassType(1); 
            } else {
                // 80% are normal HTC (Priority 0)
                long length = isHPNTS ? (long)(baseLength * (1 + OMEGA_VIRTUAL)) : baseLength;
                task = new Cloudlet(id, length, 1, 300, 300, utilizationModel, utilizationModel, utilizationModel);
                task.setUserId(brokerId);
                task.setClassType(0);
            }
            cloudletList.add(task);
        }

     // ==========================================
        // 🚀 THE HPNTS LOGIC INJECTION
        // ==========================================
        if (isHPNTS) {
            // 1. Sort by priority FIRST (so HPC tasks are at the front of the line)
            cloudletList.sort((c1, c2) -> Integer.compare(c2.getClassType(), c1.getClassType()));
        }

        // 2. Submit the list to the broker so it officially registers the IDs
        broker.submitCloudletList(cloudletList);

        // 3. Now simulate the IA3C Agent Mapping (Binding)
        if (isHPNTS) {
            int flatIndex = 0;
            int containerIndex = 0;
            
            for (Cloudlet task : cloudletList) {
                if (task.getClassType() == 1) { // HPC
                    broker.bindCloudletToVm(task.getCloudletId(), flatVms.get(flatIndex).getId());
                    flatIndex = (flatIndex + 1) % flatVms.size();
                } else { // HTC
                    broker.bindCloudletToVm(task.getCloudletId(), nestedContainers.get(containerIndex).getId());
                    containerIndex = (containerIndex + 1) % nestedContainers.size();
                }
            }
        }
        // ==========================================

        // 6. Start Simulation
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        // 7. Calculate Makespan
        List<Cloudlet> finishedList = broker.getCloudletReceivedList();
        double makespan = 0;
        for (Cloudlet cl : finishedList) {
            if (cl.getFinishTime() > makespan) {
                makespan = cl.getFinishTime();
            }
        }
        return makespan;
    }

    private static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < 8; p++) {
                peList.add(new Pe(p, new PeProvisionerSimple(5000)));
            }

            hostList.add(new Host(
                i,
                new RamProvisionerSimple(32000),
                new BwProvisionerSimple(1000000),
                100000,
                peList,
                new VmSchedulerTimeShared(peList)
            ));
        }

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.001, 0.0);

        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }

    private static void printComparison(double baseMakespan, double hpntsMakespan) {
        double makespanImprov = ((baseMakespan - hpntsMakespan) / baseMakespan) * 100;
        DecimalFormat dft = new DecimalFormat("###.##");

        System.out.println("\n=======================================================");
        System.out.println("   HPNTS PROJECT RESULTS: COMPARATIVE ANALYSIS");
        System.out.println("=======================================================");
        System.out.printf("%-20s | %-12s | %-12s | %-10s\n", "Metric", "Baseline FCFS", "HPNTS", "Improvement");
        System.out.println("-------------------------------------------------------");
        System.out.printf("%-20s | %-12s | %-12s | %s%%\n", 
            "Makespan (Seconds)", dft.format(baseMakespan), dft.format(hpntsMakespan), dft.format(makespanImprov));
        System.out.println("=======================================================");
    }
}