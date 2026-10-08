#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/nr-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/propagation-module.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

using namespace ns3;

std::ofstream dataFile;

double cell1Rsrp = 0.0;
double cell2Rsrp = 0.0;
double cell1Rsrq = 0.0;
double cell2Rsrq = 0.0;

bool handoverEvent = false;
uint16_t handoverTargetCell = 0;

void PrintUePosition(Ptr<ConstantVelocityMobilityModel> ueModel)
{
    Vector position = ueModel->GetPosition();

    std::cout << "Time " << Simulator::Now().GetSeconds()
              << " s | UE x = " << position.x
              << " m" << std::endl;
    Simulator::Schedule(Seconds(5.0), &PrintUePosition, ueModel);
}

void HandoverStartCallback(uint64_t imsi, uint16_t cellId, uint16_t rnti, uint16_t targetCellId)
{
    handoverEvent = true;
    handoverTargetCell = targetCellId;

    std::cout << "HANDOVER START"
              << " | Time = " << Simulator::Now().GetSeconds() << " s"
              << " | Source cell = " << cellId
              << " | Target cell = " << targetCellId
              << std::endl;
}

void HandoverEndOkCallback(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
       std::cout << "HANDOVER COMPLETE"
              << " | Time = " << Simulator::Now().GetSeconds() << " s"
              << " | Connect to cell = " << cellId
              << std::endl;
}

void PrintServingCell(Ptr<NrUeRrc> ueRrc)
{
    std::cout << " | Time = " << Simulator::Now().GetSeconds() 
              << "s | Serving cell = "
              << ueRrc->GetCellId()
              << std::endl;

    Simulator::Schedule(Seconds(5.0), &PrintServingCell, ueRrc);
}

void LogDatasetRow(Ptr<ConstantVelocityMobilityModel> ueModel, Ptr<NrUeRrc> ueRrc, uint32_t runId, double logInterval)
{
    double time = Simulator::Now().GetSeconds();
    Vector position = ueModel->GetPosition();
    uint16_t servingCell = ueRrc->GetCellId();

    Vector velocity = ueModel->GetVelocity();

    double speed = std::sqrt(
        velocity.x * velocity.x +
        velocity.y * velocity.y +
        velocity.z * velocity.z);
    
    double servingRsrp = 0.0;
    double neighborRsrp = 0.0;
    double servingRsrq = 0.0;
    double neighborRsrq = 0.0;

    if (servingCell == 1)
    {
        servingRsrp = cell1Rsrp;
        neighborRsrp = cell2Rsrp;

        servingRsrq = cell1Rsrq;
        neighborRsrq = cell2Rsrq;

    }
    else if (servingCell == 2)
    {
        servingRsrp = cell2Rsrp;
        neighborRsrp = cell1Rsrp;

        servingRsrq = cell2Rsrq;
        neighborRsrq = cell1Rsrq;
    }

    double rsrpDifference = neighborRsrp - servingRsrp;
    
    dataFile << runId << "," 
             << time << ","
             << position.x << ","
             << speed << ","
             << servingCell << ","
             << servingRsrp << ","
             << neighborRsrp << ","
             << servingRsrq << ","
             << neighborRsrq << ","
             << rsrpDifference << ","
             << handoverEvent << ","
             << handoverTargetCell << "\n";
    
    handoverEvent = false;
    handoverTargetCell = 0;
    
    Simulator::Schedule(Seconds(logInterval), &LogDatasetRow, ueModel, ueRrc, runId, logInterval);
}

void ReportUeMeasurementsCallback (uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool servingCell, uint8_t componentCarrierId)
{
    if (cellId == 1)
    {
        cell1Rsrp = rsrp;
        cell1Rsrq = rsrq;
    }
    else if (cellId == 2)
    {
        cell2Rsrp = rsrp;
        cell2Rsrq = rsrq;
    }
}

int main (int argc, char* argv[])
{
    double ueSpeed = 10.0;
    CommandLine cmd(__FILE__);
    std::string outputFile = "handover-data.csv";
    double startX = 50.0;
    double simTime = 60.0;
    uint32_t runId = 1;
    double logInterval = 1.0;
    double gNbDistance = 500.0;
    double gNb1TxPower = 30.0;
    double gNb2TxPower = 30.0;

    cmd.AddValue("speed", "UE movement speed in meters per second", ueSpeed);
    cmd.AddValue("output", "Name of the CSV output file", outputFile);
    cmd.AddValue("startX", "Initial UE x position in meters", startX);
    cmd.AddValue("simTime", "Simulation duration in seconds", simTime);
    cmd.AddValue("runId", "Simulation run identifier", runId);
    cmd.AddValue("logInterval", "Dataset logging interval in seconds", logInterval);
    cmd.AddValue("gNbDistance", "Distance between the two gNBs in meters", gNbDistance);
    cmd.AddValue("gNb1TxPower", "Transmit power of gNB 1 in dBm", gNb1TxPower);
    cmd.AddValue("gNb2TxPower", "Transmit power of gNB 2 in dBm", gNb2TxPower);

    cmd.Parse(argc, argv);

    std::cout << "Starting ML 5G simulation..." << std::endl;

    dataFile.open(outputFile);
    dataFile << "run_id,time_s,ue_x_m,speed_mps,serving_cell,"
             << "serving_rsrp,neighbor_rsrp,"
             << "serving_rsrq,neighbor_rsrq,"
             << "rsrp_difference,"
             << "handover_event,target_cell\n";

    NodeContainer gNbNodes; 
    NodeContainer ueNodes;

    gNbNodes.Create(2);
    ueNodes.Create(1);

    std::cout << "Number of gNBs: " << gNbNodes.GetN() << std::endl;
    std::cout << "Number of UEs: " << ueNodes.GetN() << std::endl;

    MobilityHelper gNbMobility;
    Ptr<ListPositionAllocator> gNbPositions = CreateObject<ListPositionAllocator>();

    gNbPositions->Add(Vector(0.0, 0.0, 10.0));
    gNbPositions->Add(Vector(gNbDistance, 0.0, 10.0));
    
    gNbMobility.SetPositionAllocator(gNbPositions);
    gNbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gNbMobility.Install(gNbNodes);

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    ueMobility.Install(ueNodes);

    Ptr<ConstantVelocityMobilityModel> ueModel = ueNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>();
    
    ueModel->SetPosition(Vector(startX, 0.0, 1.5));
    ueModel->SetVelocity(Vector(ueSpeed, 0.0, 0.0));

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(nrEpcHelper);

    // 5G carrier configuration
    double centralFrequency = 2.8e9;
    double bandwidth = 10e6;
    uint8_t numCcPerBand = 1;

    // Create the operation band
    CcBwpCreator ccBwpCreator;

    CcBwpCreator::SimpleOperationBandConf bandConf(
        centralFrequency,
        bandwidth,
        numCcPerBand);
    
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    // Create the radio channel 
    Ptr <NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();

    channelHelper->ConfigurePropagationFactory(FriisPropagationLossModel::GetTypeId());

    channelHelper->AssignChannelsToBands({band});

    // Get the bandwidth parts
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    nrHelper->SetHandoverAlgorithmType("ns3::NrA3RsrpHandoverAlgorithm");

    nrHelper->SetHandoverAlgorithmAttribute(
        "Hysteresis",
        DoubleValue(1.5));
    
    nrHelper->SetHandoverAlgorithmAttribute(
        "TimeToTrigger",
        TimeValue(MilliSeconds(128)));

    nrHelper->SetUeAntennaTypeId(IsotropicAntennaModel::GetTypeId().GetName());
    nrHelper->SetGnbAntennaTypeId(IsotropicAntennaModel::GetTypeId().GetName());

    // Install 5G devices
    NetDeviceContainer gNbDevices = nrHelper->InstallGnbDevice(gNbNodes, allBwps);

    NrHelper::GetGnbPhy(gNbDevices.Get(0), 0)
        ->SetAttribute("TxPower", DoubleValue(gNb1TxPower));
    
    NrHelper::GetGnbPhy(gNbDevices.Get(1), 0)
        ->SetAttribute("TxPower", DoubleValue(gNb2TxPower));

    NetDeviceContainer ueDevices = nrHelper->InstallUeDevice(ueNodes, allBwps);

    // Get the UE PHY
    Ptr<NrUePhy> uePhy = NrHelper::GetUePhy(ueDevices.Get(0), 0);

    // Listen for UE radio measurements
    uePhy->TraceConnectWithoutContext(
        "ReportUeMeasurements", 
        MakeCallback(&ReportUeMeasurementsCallback));

    // Install the Internet/IP stack on the UE
    InternetStackHelper internet;
    internet.Install(ueNodes);

    //Give the UE an IPv4 address
    Ipv4InterfaceContainer ueIpIfaces = nrEpcHelper->AssignUeIpv4Address(ueDevices);
    std::cout << "UE IP address: "
              << ueIpIfaces.GetAddress(0)
              <<std::endl;
    
    //Connect handover traces
    Ptr<NrUeNetDevice> ueNetDevice = ueDevices.Get(0)->GetObject<NrUeNetDevice>();

    Ptr<NrUeRrc> ueRrc = ueNetDevice->GetRrc();
    
    ueRrc->TraceConnectWithoutContext(
        "HandoverStart",
        MakeCallback(&HandoverStartCallback));
    
    ueRrc->TraceConnectWithoutContext(
        "HandoverEndOk",
        MakeCallback(&HandoverEndOkCallback));
    
    //Initially connect UE to gNB 0
    nrHelper->AttachToGnb(
        ueDevices.Get(0),
        gNbDevices.Get(0));
    
    //Connect the two gNBs for handover
    nrHelper->AddX2Interface(gNbNodes);

    Simulator::Schedule(Seconds(logInterval), &LogDatasetRow, ueModel,ueRrc, runId, logInterval);

    Simulator::Schedule(
    Seconds(1.0),
    &PrintServingCell,
    ueRrc);

    std::cout << "Installed gNB devices: "
              << gNbDevices.GetN() << std::endl;
    std::cout << "Installed  UE devices: "
              << ueDevices.GetN() << std::endl;

    Simulator::Schedule(Seconds(0.0), &PrintUePosition, ueModel);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();
    
    dataFile.close();

    return 0;
}