// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 AIMLAB-QuantumAI
//
// This program links against ns-3, which is distributed under the GNU General
// Public License version 2, and is distributed under the same licence; see
// ns3/LICENSE.
//
// Single-bottleneck TCP/RED scenario for ns-3: the operator's state, sampled.
//
// The classic works this scenario follows -- Hollot et al.'s controller
// design, Misra et al.'s fluid model, and the Marek/Domanski diffusion
// approximations -- each validate against ns-2/ns-3 or an independent
// discrete-event simulator. This scenario runs one operating point in ns-3
// and emits the tagged flow's (window, queue, phase) samples, from which the
// transition kernels the operators are fitted to and scored on are counted.
//
// Points about this simulator worth knowing when reading the samples:
//   * ns-3's NewReno counts real duplicate ACKs and estimates RTO adaptively.
//   * The sender is clocked by real ACK packets.
//   * RED is the Floyd-Jacobson implementation with LInterm = 1/p_max.
//
// Two load steps are available and both are off by default. Hollot et al. and
// the Marek/Domanski diffusion papers both report the transient after the
// session count changes, not only the stationary queue, because that is where
// a controller is judged.
//
//   --surgeFlows / --surgeStartS / --surgeStopS
//       extra sessions that are silent, start at the step, and stop again.
//   --releaseFlows / --releaseStopS / --releaseRestartS
//       sessions that run from the beginning, leave at the step, and rejoin.
//
// They are not mirror images. A session arriving does so in slow start and
// overshoots whatever its eventual share is, so a load increase drives this
// 90-packet buffer to its ceiling at every step size tried -- 2->4, 4->6,
// 6->8 and 2->6 all peak at 89-90. That is the system's behaviour, not a
// measurement failure, and the drop cost and settling time still separate the
// RED settings.
//
// The release is available but is not what the transient driver reports by
// default, and the reason is worth stating. Stopping a BulkSendApplication
// closes its socket gracefully, so TCP goes on delivering whatever is already
// in the send buffer. At the default 4194-packet buffer that is over a second
// of offered load and the released sessions never actually leave: a release
// run is indistinguishable from one without it (mean queue 19.13 against
// 19.13). Shrinking the buffer with --sndBufPkts does make them leave --
// at 512 packets the queue reaches the two-session level of 0.25 -- but the
// drain then takes about 1.5 s and is dominated by the buffer flush rather
// than by TCP. A departure transient measured that way would be a measurement
// of the socket buffer.
//
// With the defaults (both counts zero) the scenario is byte-for-byte the
// stationary one that produced data/ns3_validation and data/ns3_train216.
//
// Emits CSV on stdout: t_s,flow0_cwnd_pkts,queue_pkts,phase,n_active,drops
// with one row per sample interval, phase 0 = slow start, 1 = congestion
// avoidance (cwnd < ssthresh is slow start).
// n_active is the number of sessions currently sending and drops is the
// bottleneck's cumulative drop count, which the stationary panel ignores and
// the transient analysis needs.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace ns3;

namespace
{

Ptr<QueueDisc> g_bottleneckQueue;
Ptr<TcpSocketBase> g_taggedSocket;
uint32_t g_segmentSize = 1000;
double g_sampleInterval = 0.010;
double g_warmup = 8.0;
uint32_t g_samples = 1000;
uint32_t g_taken = 0;
uint32_t g_activeFlows = 0;

void
SetActiveFlows(uint32_t value)
{
    g_activeFlows = value;
}

// The tagged flow's congestion state. ns-3 reports cwnd and ssthresh in bytes;
// the grid is in packets, so both are divided by the segment size. Traced
// rather than polled because a socket attribute read mid-handshake can return
// a value the sender never acted on.
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
    const uint32_t queuePkts = g_bottleneckQueue ? g_bottleneckQueue->GetNPackets() : 0;
    const double cwndPkts = static_cast<double>(g_cwndBytes) / g_segmentSize;
    // Slow start while cwnd has not reached ssthresh (phase 0). ns-3
    // initialises ssthresh to a very large value, so a flow that has never
    // seen a loss stays in slow start.
    const int phase = (g_cwndBytes < g_ssThreshBytes) ? 0 : 1;
    const uint32_t drops =
        g_bottleneckQueue ? g_bottleneckQueue->GetStats().nTotalDroppedPackets : 0;
    std::cout << std::fixed << std::setprecision(6) << Simulator::Now().GetSeconds() << ','
              << cwndPkts << ',' << queuePkts << ',' << phase << ','
              << g_activeFlows << ',' << drops << '\n';
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
    UintegerValue value;
    g_taggedSocket->GetAttribute("SndBufSize", value);
}

} // namespace

int
main(int argc, char* argv[])
{
    // Defaults mirror configs/base.yaml's calibrated operating point.
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
    uint32_t surgeFlows = 0;
    double surgeStartS = -1.0;
    double surgeStopS = -1.0;
    uint32_t releaseFlows = 0;
    double releaseStopS = -1.0;
    double releaseRestartS = -1.0;
    uint32_t sndBufPkts = 4194;
    uint32_t burstFlows = 0;
    double burstRateFrac = 0.30;
    double burstOnMeanS = 0.10;
    double burstOffMeanS = 0.40;
    double burstShape = 1.5;

    CommandLine cmd(__FILE__);
    cmd.AddValue("capacityPps", "bottleneck capacity in packets per second", capacityPps);
    cmd.AddValue("rttPropS", "propagation round-trip time in seconds", rttPropS);
    cmd.AddValue("nFlows", "number of long-lived TCP flows", nFlows);
    cmd.AddValue("pErr", "independent packet error rate on the bottleneck", pErr);
    cmd.AddValue("pMax", "RED maximum drop probability", pMax);
    cmd.AddValue("minTh", "RED minimum queue threshold in packets", minTh);
    cmd.AddValue("maxThFrac", "RED maximum threshold as a fraction of the buffer", maxThFrac);
    cmd.AddValue("bufferPkts", "bottleneck buffer size in packets", bufferPkts);
    cmd.AddValue("qw", "RED averaging weight", qw);
    cmd.AddValue("warmup", "seconds discarded before the first sample", g_warmup);
    cmd.AddValue("samples", "number of sample intervals after the first", g_samples);
    cmd.AddValue("sampleInterval", "seconds between samples", g_sampleInterval);
    cmd.AddValue("segmentSize", "TCP segment size in bytes", g_segmentSize);
    cmd.AddValue("runSeed", "ns-3 run number", runSeed);
    cmd.AddValue("surgeFlows", "extra sessions added at the load step", surgeFlows);
    cmd.AddValue("surgeStartS",
                 "seconds after the first sample when the extra sessions start; "
                 "negative disables the step",
                 surgeStartS);
    cmd.AddValue("surgeStopS",
                 "seconds after the first sample when they stop; negative "
                 "leaves them running",
                 surgeStopS);
    cmd.AddValue("releaseFlows",
                 "sessions that run from the start and leave at the step",
                 releaseFlows);
    cmd.AddValue("releaseStopS",
                 "seconds after the first sample when they leave; negative "
                 "disables the release step",
                 releaseStopS);
    cmd.AddValue("releaseRestartS",
                 "seconds after the first sample when they rejoin; negative "
                 "leaves them gone",
                 releaseRestartS);
    cmd.AddValue("sndBufPkts",
                 "sender socket buffer in packets; only has to exceed the "
                 "congestion window, and bounds how much a departing session "
                 "still has to deliver. The default is what the stationary "
                 "panels were generated with; the largest congestion window "
                 "measured at the calibrated operating point is 58 packets, "
                 "so anything from a few hundred up cannot bind it",
                 sndBufPkts);
    cmd.AddValue("burstFlows",
                 "unresponsive UDP cross-traffic sources with Pareto on-times; "
                 "zero reproduces the long-lived-only scenario exactly",
                 burstFlows);
    cmd.AddValue("burstRateFrac",
                 "peak rate of each burst source as a fraction of the bottleneck",
                 burstRateFrac);
    cmd.AddValue("burstOnMeanS", "mean Pareto on-time in seconds", burstOnMeanS);
    cmd.AddValue("burstOffMeanS", "mean exponential off-time in seconds",
                 burstOffMeanS);
    cmd.AddValue("burstShape",
                 "Pareto shape for on-times; below 2 the aggregate is "
                 "long-range dependent, which is the regime RED was designed for",
                 burstShape);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(20260826);
    RngSeedManager::SetRun(runSeed);

    // Capacity is specified in packets per second, the unit the conditions
    // and the operator's grid use.
    const uint64_t bottleneckBps =
        static_cast<uint64_t>(capacityPps * g_segmentSize * 8.0);
    // Propagation RTT is split across the two hops, halved again because a
    // round trip crosses each link twice.
    const double oneWayEdge = rttPropS / 4.0;
    const double oneWayCore = rttPropS / 4.0;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpNewReno")));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(false));
    // Large enough that the application never becomes the binding constraint;
    // the point of the scenario is the bottleneck queue, not sender buffering.
    //
    // It must not be *much* larger than that either, and the release step is
    // what exposed why. A BulkSendApplication fills this buffer as fast as the
    // socket accepts, and stopping the application only closes the socket
    // gracefully: TCP goes on delivering everything already buffered. At the
    // original 4 MB that is 4194 packets, several seconds of offered load at
    // this capacity, so a released session kept sending and a release run was
    // indistinguishable from one with no release at all -- 25.12 mean packets
    // either way. The default here still exceeds the 64-packet maximum window
    // by 65x, so it cannot bind the congestion window.
    Config::SetDefault("ns3::TcpSocket::SndBufSize",
                       UintegerValue(sndBufPkts * g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 22));

    // Surge sessions exist from the start but do not send until the step, so
    // the topology is fixed and only the offered load changes. Building nodes
    // mid-run would also change the routing tables, which is not the
    // perturbation Hollot's step response is about.
    const bool stepEnabled = surgeFlows > 0 && surgeStartS >= 0.0;
    const bool releaseEnabled = releaseFlows > 0 && releaseStopS >= 0.0;
    const uint32_t totalFlows = nFlows + (stepEnabled ? surgeFlows : 0)
                                + (releaseEnabled ? releaseFlows : 0);

    NodeContainer senders;
    senders.Create(totalFlows);
    NodeContainer routers;
    routers.Create(2);
    NodeContainer receivers;
    receivers.Create(totalFlows);

    PointToPointHelper edge;
    edge.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bottleneckBps * 10)));
    edge.SetChannelAttribute("Delay", TimeValue(Seconds(oneWayEdge)));
    // Edge links must never be the bottleneck, so give them a deep FIFO.
    edge.SetQueue("ns3::DropTailQueue<Packet>",
                  "MaxSize", QueueSizeValue(QueueSize("1000p")));

    PointToPointHelper core;
    core.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bottleneckBps)));
    core.SetChannelAttribute("Delay", TimeValue(Seconds(oneWayCore)));
    core.SetQueue("ns3::DropTailQueue<Packet>",
                  "MaxSize", QueueSizeValue(QueueSize("1p")));

    NetDeviceContainer coreDevices = core.Install(routers.Get(0), routers.Get(1));

    if (pErr > 0.0)
    {
        Ptr<RateErrorModel> error = CreateObject<RateErrorModel>();
        error->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
        error->SetRate(pErr);
        coreDevices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(error));
    }

    InternetStackHelper stack;
    stack.Install(senders);
    stack.Install(routers);
    stack.Install(receivers);

    // RED on the bottleneck only. LInterm is the reciprocal of the maximum
    // drop probability in ns-3's Floyd-Jacobson implementation.
    //
    // Gentle mode is not optional here. The operator's RED model ramps from
    // p_max at max_th linearly to 1 at 2 * max_th: a hard drop-all at max_th
    // would leave the tail region empty and make the overflow probability
    // unmeasurable. Since q_tail = 0.8 * buffer = 72 packets sits above
    // max_th = 45, hard-drop RED confines the queue below the very region of
    // interest -- a first run with UseHardDrop never left queue bins 0-5 of 16.
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
    QueueDiscContainer coreQueues = red.Install(coreDevices);
    g_bottleneckQueue = coreQueues.Get(0);

    // The reverse direction carries only ACKs; a deep FIFO keeps it clear.
    TrafficControlHelper fifo;
    fifo.SetRootQueueDisc("ns3::FifoQueueDisc",
                          "MaxSize", QueueSizeValue(QueueSize("1000p")));

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.255.0");
    Ipv4InterfaceContainer coreInterfaces = address.Assign(coreDevices);

    ApplicationContainer sinks;
    ApplicationContainer sources;
    const uint16_t port = 50000;
    std::vector<Ipv4Address> receiverAddresses;
    receiverAddresses.reserve(totalFlows);

    for (uint32_t i = 0; i < totalFlows; ++i)
    {
        NetDeviceContainer left = edge.Install(senders.Get(i), routers.Get(0));
        NetDeviceContainer right = edge.Install(routers.Get(1), receivers.Get(i));
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

        receiverAddresses.push_back(rightInterfaces.GetAddress(1));
        BulkSendHelper source("ns3::TcpSocketFactory",
                              InetSocketAddress(rightInterfaces.GetAddress(1), port));
        source.SetAttribute("MaxBytes", UintegerValue(0));
        sources.Add(source.Install(senders.Get(i)));
    }

    // Unresponsive bursty cross traffic. The stationary datasets offer only
    // long-lived bulk TCP, which is the one traffic class RED was not designed
    // for: Floyd and Jacobson motivate RED by the need to absorb bursts, and
    // the diffusion-approximation papers drive
    // their queues with heavy-tailed arrivals rather than with a fixed number
    // of greedy senders. Pareto on-times with shape below two make the
    // aggregate long-range dependent, which is the standard construction.
    //
    // UDP rather than TCP, and deliberately so. The point is traffic the
    // congestion controller cannot absorb by backing off, because that is what
    // makes the queue distribution non-trivial. TCP cross traffic would mostly
    // reproduce the flow-count axis the panel already varies.
    if (burstFlows > 0)
    {
        NodeContainer burstSenders;
        burstSenders.Create(burstFlows);
        NodeContainer burstReceivers;
        burstReceivers.Create(burstFlows);
        stack.Install(burstSenders);
        stack.Install(burstReceivers);

        const uint16_t burstPort = 60000;
        const uint64_t burstBps =
            static_cast<uint64_t>(burstRateFrac * bottleneckBps);
        // Pareto mean is scale * shape / (shape - 1) for shape > 1.
        const double onScale =
            burstOnMeanS * (burstShape - 1.0) / burstShape;

        for (uint32_t i = 0; i < burstFlows; ++i)
        {
            NetDeviceContainer left = edge.Install(burstSenders.Get(i), routers.Get(0));
            NetDeviceContainer right = edge.Install(routers.Get(1), burstReceivers.Get(i));
            fifo.Install(left);
            fifo.Install(right);

            std::ostringstream leftBase;
            leftBase << "10.4." << i << ".0";
            address.SetBase(leftBase.str().c_str(), "255.255.255.0");
            address.Assign(left);

            std::ostringstream rightBase;
            rightBase << "10.5." << i << ".0";
            address.SetBase(rightBase.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer rightInterfaces = address.Assign(right);

            PacketSinkHelper burstSink(
                "ns3::UdpSocketFactory",
                InetSocketAddress(Ipv4Address::GetAny(), burstPort));
            sinks.Add(burstSink.Install(burstReceivers.Get(i)));

            OnOffHelper burst("ns3::UdpSocketFactory",
                              InetSocketAddress(rightInterfaces.GetAddress(1),
                                                burstPort));
            burst.SetAttribute("DataRate", DataRateValue(DataRate(burstBps)));
            burst.SetAttribute("PacketSize", UintegerValue(g_segmentSize));
            std::ostringstream onStream;
            onStream << "ns3::ParetoRandomVariable[Scale=" << onScale
                     << "|Shape=" << burstShape << "]";
            std::ostringstream offStream;
            offStream << "ns3::ExponentialRandomVariable[Mean=" << burstOffMeanS
                      << "]";
            burst.SetAttribute("OnTime", StringValue(onStream.str()));
            burst.SetAttribute("OffTime", StringValue(offStream.str()));
            ApplicationContainer app = burst.Install(burstSenders.Get(i));
            // Offset so the sources do not switch on together, which would be
            // a synchronised square wave rather than bursty traffic.
            app.Start(Seconds(0.02 * i));
        }
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    sinks.Start(Seconds(0.0));
    // Staggering the starts by a small amount avoids a synchronised
    // slow-start burst that would put every flow in lockstep.
    const uint32_t alwaysOn = nFlows + (releaseEnabled ? releaseFlows : 0);
    for (uint32_t i = 0; i < alwaysOn; ++i)
    {
        sources.Get(i)->SetStartTime(Seconds(0.01 * i));
    }
    g_activeFlows = alwaysOn;

    if (releaseEnabled)
    {
        const double leave = g_warmup + releaseStopS;
        for (uint32_t i = nFlows; i < alwaysOn; ++i)
        {
            sources.Get(i)->SetStopTime(Seconds(leave + 0.01 * (i - nFlows)));
        }
        Simulator::Schedule(Seconds(leave), &SetActiveFlows, nFlows);
        if (releaseRestartS >= 0.0)
        {
            // A BulkSendApplication that has stopped cannot be started again,
            // so the rejoin is a second application on the same node. That is
            // also the truthful behaviour: a returning session opens a new
            // connection and enters slow start rather than resuming its old
            // window. Scheduling only the SetActiveFlows label without this
            // left the trace identical to a run with no rejoin at all.
            const double rejoin = g_warmup + releaseRestartS;
            for (uint32_t i = nFlows; i < alwaysOn; ++i)
            {
                BulkSendHelper again(
                    "ns3::TcpSocketFactory",
                    InetSocketAddress(receiverAddresses[i], port));
                again.SetAttribute("MaxBytes", UintegerValue(0));
                ApplicationContainer app = again.Install(senders.Get(i));
                app.Start(Seconds(rejoin + 0.01 * (i - nFlows)));
            }
            Simulator::Schedule(Seconds(rejoin), &SetActiveFlows, alwaysOn);
        }
    }

    if (stepEnabled)
    {
        // Step times are given relative to the first sample so a trace can be
        // aligned without knowing the warmup, which differs between runs.
        const double surgeStart = g_warmup + surgeStartS;
        for (uint32_t i = alwaysOn; i < totalFlows; ++i)
        {
            sources.Get(i)->SetStartTime(Seconds(surgeStart + 0.01 * (i - alwaysOn)));
            if (surgeStopS >= 0.0)
            {
                sources.Get(i)->SetStopTime(Seconds(g_warmup + surgeStopS));
            }
        }
        Simulator::Schedule(Seconds(surgeStart), &SetActiveFlows, totalFlows);
        if (surgeStopS >= 0.0)
        {
            Simulator::Schedule(Seconds(g_warmup + surgeStopS), &SetActiveFlows,
                                alwaysOn);
        }
    }

    // Flow 0 is the tagged flow whose state is sampled.
    Simulator::Schedule(Seconds(0.05), &ConnectTaggedTraces, sources.Get(0));
    Simulator::Schedule(Seconds(g_warmup), &Sample);

    const double stop = g_warmup + (g_samples + 2) * g_sampleInterval;
    Simulator::Stop(Seconds(stop));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
