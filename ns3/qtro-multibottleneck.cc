// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 AIMLAB-QuantumAI
//
// This program links against ns-3, which is distributed under the GNU General
// Public License version 2, and is distributed under the same licence; see
// ns3/LICENSE.
//
// Two RED bottlenecks in series, for ns-3.
//
// Misra, Gong and Towsley validate their fluid model on a *network* of AQM
// routers, not only on one dumbbell, and the tandem case is where a
// single-queue model has something to lose: the tagged flow's arrivals into
// the second queue have already been shaped by the first, so the second
// bottleneck's dynamics are not what any single-bottleneck operator was
// conditioned on.
//
// The stationary datasets are single dumbbells. This scenario keeps the
// measured quantities and the sampling identical to qtro-dumbbell.cc and
// changes only the topology:
//
//   senders --- R0 ==[RED 1]== R1 ==[RED 2]== R2 --- receivers
//                               |
//                        cross traffic in/out
//
// The tagged flow traverses both. Cross traffic enters at R1 and leaves at R2,
// so it loads the second bottleneck only. The emitted queue is the *first*
// bottleneck's, which is the one the RED action in u applies to, so the
// experiment asks whether an operator conditioned on one queue still predicts
// that queue when a second one downstream is shaping the same flow.
//
// The second bottleneck's capacity is given as a fraction of the first. At
// --capacityRatio 1.0 with no cross traffic the second queue almost never
// fills and the scenario reduces to the dumbbell; below 1.0, or with cross
// traffic, the second queue becomes the binding constraint for part of the
// time and the tagged flow's window is governed by something the operator
// cannot see.
//
// Emits the same CSV as qtro-dumbbell.cc:
//   t_s,flow0_cwnd_pkts,queue_pkts,phase,n_active,drops
// where queue_pkts and drops are the first bottleneck's, plus two extra
// columns q2_pkts,drops2 for the second, which the shift analysis uses to say
// whether the second queue was actually active.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace ns3;

namespace
{

Ptr<QueueDisc> g_queue1;
Ptr<QueueDisc> g_queue2;
Ptr<TcpSocketBase> g_taggedSocket;
uint32_t g_segmentSize = 1000;
double g_sampleInterval = 0.010;
double g_warmup = 8.0;
uint32_t g_samples = 1000;
uint32_t g_taken = 0;
uint32_t g_activeFlows = 0;

uint32_t g_cwndBytes = 0;
uint32_t g_ssThreshBytes = 0;

void
CwndTracer(uint32_t, uint32_t next)
{
    g_cwndBytes = next;
}

void
SsThreshTracer(uint32_t, uint32_t next)
{
    g_ssThreshBytes = next;
}

void
Sample()
{
    if (g_taken >= g_samples + 1)
    {
        return;
    }
    const uint32_t q1 = g_queue1 ? g_queue1->GetNPackets() : 0;
    const uint32_t q2 = g_queue2 ? g_queue2->GetNPackets() : 0;
    const double cwndPkts = static_cast<double>(g_cwndBytes) / g_segmentSize;
    const int phase = (g_cwndBytes < g_ssThreshBytes) ? 0 : 1;
    const uint32_t d1 = g_queue1 ? g_queue1->GetStats().nTotalDroppedPackets : 0;
    const uint32_t d2 = g_queue2 ? g_queue2->GetStats().nTotalDroppedPackets : 0;
    std::cout << std::fixed << std::setprecision(6) << Simulator::Now().GetSeconds() << ','
              << cwndPkts << ',' << q1 << ',' << phase << ',' << g_activeFlows << ','
              << d1 << ',' << q2 << ',' << d2 << '\n';
    ++g_taken;
    Simulator::Schedule(Seconds(g_sampleInterval), &Sample);
}

void
ConnectTaggedTraces(Ptr<Application> app)
{
    Ptr<BulkSendApplication> bulk = DynamicCast<BulkSendApplication>(app);
    if (!bulk)
    {
        return;
    }
    Ptr<Socket> socket = bulk->GetSocket();
    g_taggedSocket = DynamicCast<TcpSocketBase>(socket);
    if (!g_taggedSocket)
    {
        return;
    }
    g_taggedSocket->TraceConnectWithoutContext("CongestionWindow",
                                               MakeCallback(&CwndTracer));
    g_taggedSocket->TraceConnectWithoutContext("SlowStartThreshold",
                                               MakeCallback(&SsThreshTracer));
}

} // namespace

int
main(int argc, char* argv[])
{
    double capacityPps = 2000.0;
    double rttPropS = 0.030;
    uint32_t nFlows = 4;
    double pErr = 0.0;
    double pMax = 0.1;
    double minTh = 20.0;
    double maxThFrac = 0.5;
    double bufferPkts = 90.0;
    double qw = 1e-4;
    uint32_t runSeed = 1;
    uint32_t sndBufPkts = 4194;
    double capacityRatio = 1.0;
    uint32_t crossFlows = 0;
    double crossRateFrac = 0.40;
    double crossOnMeanS = 0.10;
    double crossOffMeanS = 0.40;
    double crossShape = 1.5;

    CommandLine cmd(__FILE__);
    cmd.AddValue("capacityPps", "first bottleneck capacity in packets per second",
                 capacityPps);
    cmd.AddValue("rttPropS", "propagation round-trip time in seconds", rttPropS);
    cmd.AddValue("nFlows", "number of long-lived TCP flows crossing both hops",
                 nFlows);
    cmd.AddValue("pErr", "independent packet error rate on the first bottleneck",
                 pErr);
    cmd.AddValue("pMax", "RED maximum drop probability, both queues", pMax);
    cmd.AddValue("minTh", "RED minimum queue threshold in packets, both queues",
                 minTh);
    cmd.AddValue("maxThFrac", "RED maximum threshold as a fraction of the buffer",
                 maxThFrac);
    cmd.AddValue("bufferPkts", "buffer size in packets, both queues", bufferPkts);
    cmd.AddValue("qw", "RED averaging weight", qw);
    cmd.AddValue("warmup", "seconds discarded before the first sample", g_warmup);
    cmd.AddValue("samples", "number of sample intervals after the first", g_samples);
    cmd.AddValue("sampleInterval", "seconds between samples", g_sampleInterval);
    cmd.AddValue("segmentSize", "TCP segment size in bytes", g_segmentSize);
    cmd.AddValue("runSeed", "ns-3 run number", runSeed);
    cmd.AddValue("sndBufPkts", "sender socket buffer in packets", sndBufPkts);
    cmd.AddValue("capacityRatio",
                 "second bottleneck capacity as a fraction of the first",
                 capacityRatio);
    cmd.AddValue("crossFlows",
                 "bursty UDP sources loading the second bottleneck only",
                 crossFlows);
    cmd.AddValue("crossRateFrac",
                 "peak rate of each cross source as a fraction of the second hop",
                 crossRateFrac);
    cmd.AddValue("crossOnMeanS", "mean Pareto on-time in seconds", crossOnMeanS);
    cmd.AddValue("crossOffMeanS", "mean exponential off-time in seconds",
                 crossOffMeanS);
    cmd.AddValue("crossShape", "Pareto shape for cross on-times", crossShape);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(20260826);
    RngSeedManager::SetRun(runSeed);

    const uint64_t bps1 = static_cast<uint64_t>(capacityPps * g_segmentSize * 8.0);
    const uint64_t bps2 = static_cast<uint64_t>(capacityRatio * capacityPps *
                                                g_segmentSize * 8.0);
    // The propagation RTT is split over three hops out and three back, so each
    // one-way hop carries a sixth. At capacityRatio 1 and no cross traffic this
    // keeps the tagged flow's RTT equal to the dumbbell's.
    const double oneWay = rttPropS / 6.0;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpNewReno")));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::SndBufSize",
                       UintegerValue(sndBufPkts * g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 22));

    NodeContainer senders;
    senders.Create(nFlows);
    NodeContainer routers;
    routers.Create(3);
    NodeContainer receivers;
    receivers.Create(nFlows);
    NodeContainer crossSenders;
    NodeContainer crossReceivers;
    if (crossFlows > 0)
    {
        crossSenders.Create(crossFlows);
        crossReceivers.Create(crossFlows);
    }

    PointToPointHelper edge;
    edge.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bps1 * 10)));
    edge.SetChannelAttribute("Delay", TimeValue(Seconds(oneWay)));
    edge.SetQueue("ns3::DropTailQueue<Packet>",
                  "MaxSize", QueueSizeValue(QueueSize("1000p")));

    PointToPointHelper core1;
    core1.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bps1)));
    core1.SetChannelAttribute("Delay", TimeValue(Seconds(oneWay)));
    core1.SetQueue("ns3::DropTailQueue<Packet>",
                   "MaxSize", QueueSizeValue(QueueSize("1p")));

    PointToPointHelper core2;
    core2.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bps2)));
    core2.SetChannelAttribute("Delay", TimeValue(Seconds(oneWay)));
    core2.SetQueue("ns3::DropTailQueue<Packet>",
                   "MaxSize", QueueSizeValue(QueueSize("1p")));

    NetDeviceContainer core1Devices = core1.Install(routers.Get(0), routers.Get(1));
    NetDeviceContainer core2Devices = core2.Install(routers.Get(1), routers.Get(2));

    if (pErr > 0.0)
    {
        Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
        error->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
        error->SetRate(pErr);
        core1Devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(error));
    }

    InternetStackHelper stack;
    stack.Install(senders);
    stack.Install(routers);
    stack.Install(receivers);
    if (crossFlows > 0)
    {
        stack.Install(crossSenders);
        stack.Install(crossReceivers);
    }

    // Both bottlenecks carry the same gentle RED configuration, so the second
    // queue is not a different mechanism, only a second instance of the same
    // one at a place the operator's state does not describe.
    TrafficControlHelper red;
    red.SetRootQueueDisc("ns3::RedQueueDisc",
                         "MinTh", DoubleValue(minTh),
                         "MaxTh", DoubleValue(maxThFrac * bufferPkts),
                         "LInterm", DoubleValue(1.0 / pMax),
                         "QW", DoubleValue(qw),
                         "MaxSize",
                         QueueSizeValue(QueueSize(QueueSizeUnit::PACKETS,
                                                  static_cast<uint32_t>(bufferPkts))),
                         "UseEcn", BooleanValue(false),
                         "Gentle", BooleanValue(true),
                         "UseHardDrop", BooleanValue(false),
                         "MeanPktSize", UintegerValue(g_segmentSize));
    QueueDiscContainer q1 = red.Install(core1Devices);
    QueueDiscContainer q2 = red.Install(core2Devices);
    g_queue1 = q1.Get(0);
    g_queue2 = q2.Get(0);

    TrafficControlHelper fifo;
    fifo.SetRootQueueDisc("ns3::FifoQueueDisc",
                          "MaxSize", QueueSizeValue(QueueSize("1000p")));

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.255.0");
    address.Assign(core1Devices);
    address.SetBase("10.6.0.0", "255.255.255.0");
    address.Assign(core2Devices);

    ApplicationContainer sinks;
    ApplicationContainer sources;
    const uint16_t port = 50000;

    for (uint32_t i = 0; i < nFlows; ++i)
    {
        NetDeviceContainer left = edge.Install(senders.Get(i), routers.Get(0));
        NetDeviceContainer right = edge.Install(routers.Get(2), receivers.Get(i));
        fifo.Install(left);
        fifo.Install(right);

        std::ostringstream leftBase;
        leftBase << "10.2." << i << ".0";
        address.SetBase(leftBase.str().c_str(), "255.255.255.0");
        address.Assign(left);

        std::ostringstream rightBase;
        rightBase << "10.3." << i << ".0";
        address.SetBase(rightBase.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer rightInterfaces = address.Assign(right);

        PacketSinkHelper sink("ns3::TcpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        sinks.Add(sink.Install(receivers.Get(i)));

        BulkSendHelper source("ns3::TcpSocketFactory",
                              InetSocketAddress(rightInterfaces.GetAddress(1), port));
        source.SetAttribute("MaxBytes", UintegerValue(0));
        sources.Add(source.Install(senders.Get(i)));
    }

    // Cross traffic joins at R1 and leaves at R2, so it shares the second
    // bottleneck with the tagged flow and never touches the first.
    //
    // It is bursty UDP rather than long-lived TCP, and that is forced by the
    // physics of a tandem: only the tightest link queues. Steady cross traffic
    // downstream makes the second hop tighter at all times, the tagged flow
    // then backs off before the first hop fills, and the queue this scenario
    // reports collapses to zero -- measured at 16 occupied cells against the
    // dumbbell's 72, which is too few to estimate a kernel from. With Pareto
    // on/off sources the binding constraint alternates: the first hop queues
    // between bursts and the second takes over during them, which is the
    // perturbation the experiment is about.
    const uint16_t crossPort = 60000;
    const uint64_t crossBps = static_cast<uint64_t>(crossRateFrac * bps2);
    const double crossOnScale = crossOnMeanS * (crossShape - 1.0) / crossShape;
    for (uint32_t i = 0; i < crossFlows; ++i)
    {
        NetDeviceContainer left = edge.Install(crossSenders.Get(i), routers.Get(1));
        NetDeviceContainer right = edge.Install(routers.Get(2), crossReceivers.Get(i));
        fifo.Install(left);
        fifo.Install(right);

        std::ostringstream leftBase;
        leftBase << "10.7." << i << ".0";
        address.SetBase(leftBase.str().c_str(), "255.255.255.0");
        address.Assign(left);

        std::ostringstream rightBase;
        rightBase << "10.8." << i << ".0";
        address.SetBase(rightBase.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer rightInterfaces = address.Assign(right);

        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), crossPort));
        sinks.Add(sink.Install(crossReceivers.Get(i)));

        OnOffHelper cross("ns3::UdpSocketFactory",
                          InetSocketAddress(rightInterfaces.GetAddress(1),
                                            crossPort));
        cross.SetAttribute("DataRate", DataRateValue(DataRate(crossBps)));
        cross.SetAttribute("PacketSize", UintegerValue(g_segmentSize));
        std::ostringstream onStream;
        onStream << "ns3::ParetoRandomVariable[Scale=" << crossOnScale
                 << "|Shape=" << crossShape << "]";
        std::ostringstream offStream;
        offStream << "ns3::ExponentialRandomVariable[Mean=" << crossOffMeanS << "]";
        cross.SetAttribute("OnTime", StringValue(onStream.str()));
        cross.SetAttribute("OffTime", StringValue(offStream.str()));
        ApplicationContainer app = cross.Install(crossSenders.Get(i));
        app.Start(Seconds(0.02 * i));
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    sinks.Start(Seconds(0.0));
    for (uint32_t i = 0; i < sources.GetN(); ++i)
    {
        sources.Get(i)->SetStartTime(Seconds(0.01 * i));
    }
    g_activeFlows = nFlows;

    Simulator::Schedule(Seconds(0.05), &ConnectTaggedTraces, sources.Get(0));
    Simulator::Schedule(Seconds(g_warmup), &Sample);

    const double stop = g_warmup + (g_samples + 2) * g_sampleInterval;
    Simulator::Stop(Seconds(stop));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
