// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 AIMLAB-QuantumAI
//
// This program links against ns-3, which is distributed under the GNU General
// Public License version 2, and is distributed under the same licence; see
// ns3/LICENSE.
//
// ReconfigurableRed below is derived from ns-3's RedQueueDisc
// (src/traffic-control/model/red-queue-disc.cc), which carries this notice:
//
/*
 * Copyright © 2011 Marcos Talau
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Marcos Talau (talau@users.sourceforge.net)
 *
 * Thanks to: Duy Nguyen<duy@soe.ucsc.edu> by RED efforts in NS3
 *
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 * Copyright (c) 1990-1997 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor of the Laboratory may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
//
// Closed-loop TCP/RED scenario: ns-3 emits state, an external controller
// answers with a RED setting, once per control interval.
//
// Hollot et al.'s paper is entirely about the loop -- linearize, design a
// controller, and judge it by the queue it produces under load. The stationary
// datasets are open loop: a RED setting is fixed for a whole run and the
// operator is scored on prediction alone. This scenario supplies the loop,
// so a predictor can be judged by the queue its decisions produce rather than
// only by its fidelity.
//
// Two problems had to be solved, and both are worth stating because they
// constrain what the result means.
//
// 1. ns-3's RedQueueDisc cannot be reconfigured while it runs. MinTh, MaxTh
//    and LInterm are settable attributes, but the quantities the drop
//    probability is actually computed from -- m_vA, m_vB, m_vC, m_vD and
//    m_curMaxP -- are private and derived once in InitializeParams(), which is
//    itself private. Calling it again would also reset m_qAvg and m_count, so
//    even reaching it would destroy the averaged queue the controller is
//    supposed to be acting on.
//
//    So this file carries its own queue disc. It is not a new AQM: it is
//    ns-3's gentle RED, restricted to the configuration the experiments use
//    (Gentle, no ECN, no hard drop, packet mode, Wait on, no ARED, no Feng
//    adaptation, no NLRED, no ns-1 compatibility, Cautious 0), with the
//    derived constants recomputed on demand and the averaged-queue state left
//    alone. --useStockRed runs ns-3's own class instead, and with a fixed RNG
//    stream the two produce byte-identical traces when no action is applied.
//    That equivalence is the licence for using the reconfigurable one.
//
// 2. The controller is a PyTorch model, so it lives outside ns-3. At each
//    control instant the scenario writes one CTRL line to stdout and blocks
//    reading one line back from stdin. ns-3 is single threaded and event
//    driven, so a blocking read inside a scheduled event stops simulated time
//    exactly where it should.
//
// Protocol, line oriented, both directions:
//   out  CTRL,<t_s>,<qAvg>,<qInst>,<cwndPkts>,<phase>,<drops>,<deliveredPkts>,<nActive>
//   in   <pMax>,<minTh>
// A controller that wants to change nothing echoes the current values. On EOF
// the scenario keeps the last setting and runs to completion, which is what
// makes an uncontrolled reference run possible through the same code path.
//
// Sample rows are emitted as in qtro-dumbbell.cc and prefixed to keep the two
// streams separable:
//   S,t_s,cwnd_pkts,queue_pkts,phase,n_active,drops

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

// ---------------------------------------------------------------- queue disc

// Gentle RED with the derived constants recomputed on demand.
//
// Every line of the enqueue path below is ns-3's, with the branches that our
// configuration never takes removed rather than reimplemented. Keeping the
// dead branches would not make it more faithful, only harder to check against
// the original.
class ReconfigurableRed : public QueueDisc
{
  public:
    static TypeId GetTypeId();

    ReconfigurableRed();
    ~ReconfigurableRed() override = default;

    int64_t AssignStreams(int64_t stream);

    // The whole point of the class. Recomputes exactly the constants
    // InitializeParams() derives, and touches no averaged state.
    void Reconfigure(double pMax, double minTh);

    double GetQAvg() const
    {
        return m_qAvg;
    }

    double GetPMax() const
    {
        return m_curMaxP;
    }

    double GetMinTh() const
    {
        return m_minTh;
    }

  private:
    bool DoEnqueue(Ptr<QueueDiscItem> item) override;
    Ptr<QueueDiscItem> DoDequeue() override;
    Ptr<const QueueDiscItem> DoPeek() override;
    bool CheckConfig() override;
    void InitializeParams() override;

    double Estimator(uint32_t nQueued, uint32_t m, double qAvg, double qW) const;
    bool DropEarly(Ptr<QueueDiscItem> item, uint32_t qSize);
    double CalculatePNew() const;
    double ModifyP(double p) const;
    void Derive();

    // Configuration.
    double m_minTh{5.0};
    double m_maxTh{15.0};
    double m_qW{0.002};
    double m_lInterm{50.0};
    uint32_t m_meanPktSize{500};
    DataRate m_linkBandwidth{DataRate("1.5Mbps")};

    // Derived, recomputed by Derive().
    double m_vA{0.0};
    double m_vB{0.0};
    double m_vC{0.0};
    double m_vD{0.0};
    double m_curMaxP{0.0};
    double m_ptc{0.0};

    // Running state, never reset by Reconfigure().
    double m_qAvg{0.0};
    uint32_t m_count{0};
    uint32_t m_old{0};
    uint32_t m_idle{1};
    double m_vProb{0.0};
    Time m_idleTime{NanoSeconds(0)};

    Ptr<UniformRandomVariable> m_uv;
};

NS_OBJECT_ENSURE_REGISTERED(ReconfigurableRed);

TypeId
ReconfigurableRed::GetTypeId()
{
    static TypeId tid =
        TypeId("ReconfigurableRed")
            .SetParent<QueueDisc>()
            .SetGroupName("TrafficControl")
            .AddConstructor<ReconfigurableRed>()
            .AddAttribute("MinTh", "Minimum average length threshold in packets",
                          DoubleValue(5),
                          MakeDoubleAccessor(&ReconfigurableRed::m_minTh),
                          MakeDoubleChecker<double>())
            .AddAttribute("MaxTh", "Maximum average length threshold in packets",
                          DoubleValue(15),
                          MakeDoubleAccessor(&ReconfigurableRed::m_maxTh),
                          MakeDoubleChecker<double>())
            .AddAttribute("QW", "Queue weight",
                          DoubleValue(0.002),
                          MakeDoubleAccessor(&ReconfigurableRed::m_qW),
                          MakeDoubleChecker<double>())
            .AddAttribute("LInterm", "The maximum probability of dropping a packet",
                          DoubleValue(50),
                          MakeDoubleAccessor(&ReconfigurableRed::m_lInterm),
                          MakeDoubleChecker<double>())
            .AddAttribute("MeanPktSize", "Average of packet size",
                          UintegerValue(500),
                          MakeUintegerAccessor(&ReconfigurableRed::m_meanPktSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("LinkBandwidth", "The RED link bandwidth",
                          DataRateValue(DataRate("1.5Mbps")),
                          MakeDataRateAccessor(&ReconfigurableRed::m_linkBandwidth),
                          MakeDataRateChecker())
            .AddAttribute("MaxSize", "The maximum number of packets accepted",
                          QueueSizeValue(QueueSize("25p")),
                          MakeQueueSizeAccessor(&QueueDisc::SetMaxSize,
                                                &QueueDisc::GetMaxSize),
                          MakeQueueSizeChecker());
    return tid;
}

ReconfigurableRed::ReconfigurableRed()
    : QueueDisc(QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE)
{
    m_uv = CreateObject<UniformRandomVariable>();
}

int64_t
ReconfigurableRed::AssignStreams(int64_t stream)
{
    m_uv->SetStream(stream);
    return 1;
}

void
ReconfigurableRed::Derive()
{
    double thDiff = m_maxTh - m_minTh;
    if (thDiff == 0)
    {
        thDiff = 1.0;
    }
    m_vA = 1.0 / thDiff;
    m_curMaxP = 1.0 / m_lInterm;
    m_vB = -m_minTh / thDiff;
    // Gentle mode only; the non-gentle constants are never read here.
    m_vC = (1.0 - m_curMaxP) / m_maxTh;
    m_vD = 2.0 * m_curMaxP - 1.0;
}

void
ReconfigurableRed::InitializeParams()
{
    m_ptc = m_linkBandwidth.GetBitRate() / (8.0 * m_meanPktSize);
    NS_ASSERT(m_minTh <= m_maxTh);
    m_qAvg = 0.0;
    m_count = 0;
    m_old = 0;
    m_idle = 1;
    Derive();
    m_idleTime = NanoSeconds(0);
}

void
ReconfigurableRed::Reconfigure(double pMax, double minTh)
{
    m_lInterm = 1.0 / pMax;
    m_minTh = minTh;
    NS_ASSERT(m_minTh <= m_maxTh);
    Derive();
}

double
ReconfigurableRed::Estimator(uint32_t nQueued, uint32_t m, double qAvg,
                             double qW) const
{
    double newAve = qAvg * std::pow(1.0 - qW, m);
    newAve += qW * nQueued;
    return newAve;
}

double
ReconfigurableRed::CalculatePNew() const
{
    double p;
    if (m_qAvg >= m_maxTh)
    {
        p = m_vC * m_qAvg + m_vD;
    }
    else
    {
        p = m_vA * m_qAvg + m_vB;
        p *= m_curMaxP;
    }
    if (p > 1.0)
    {
        p = 1.0;
    }
    return p;
}

double
ReconfigurableRed::ModifyP(double p) const
{
    // Wait is on, packet mode, so count is the packet count.
    const auto count1 = static_cast<double>(m_count);
    if (count1 * p < 1.0)
    {
        p = 0.0;
    }
    else if (count1 * p < 2.0)
    {
        p /= (2.0 - count1 * p);
    }
    else
    {
        p = 1.0;
    }
    if (p > 1.0)
    {
        p = 1.0;
    }
    return p;
}

bool
ReconfigurableRed::DropEarly(Ptr<QueueDiscItem> item, uint32_t qSize)
{
    double prob1 = CalculatePNew();
    m_vProb = ModifyP(prob1);
    double u = m_uv->GetValue();
    if (u <= m_vProb)
    {
        m_count = 0;
        return true;
    }
    return false;
}

bool
ReconfigurableRed::DoEnqueue(Ptr<QueueDiscItem> item)
{
    uint32_t nQueued = GetInternalQueue(0)->GetCurrentSize().GetValue();

    uint32_t m = 0;
    if (m_idle == 1)
    {
        Time now = Simulator::Now();
        m = static_cast<uint32_t>(m_ptc * (now - m_idleTime).GetSeconds());
        m_idle = 0;
    }

    m_qAvg = Estimator(nQueued, m + 1, m_qAvg, m_qW);
    m_count++;

    bool forced = false;
    bool unforced = false;
    if (m_qAvg >= m_minTh && nQueued > 1)
    {
        if (m_qAvg >= 2 * m_maxTh)
        {
            forced = true;
        }
        else if (m_old == 0)
        {
            m_count = 1;
            m_old = 1;
        }
        else if (DropEarly(item, nQueued))
        {
            unforced = true;
        }
    }
    else
    {
        m_vProb = 0.0;
        m_old = 0;
    }

    if (unforced)
    {
        DropBeforeEnqueue(item, "Unforced drop");
        return false;
    }
    if (forced)
    {
        DropBeforeEnqueue(item, "Forced drop");
        return false;
    }

    bool retval = GetInternalQueue(0)->Enqueue(item);
    return retval;
}

Ptr<QueueDiscItem>
ReconfigurableRed::DoDequeue()
{
    if (GetInternalQueue(0)->IsEmpty())
    {
        m_idle = 1;
        m_idleTime = Simulator::Now();
        return nullptr;
    }
    m_idle = 0;
    return GetInternalQueue(0)->Dequeue();
}

Ptr<const QueueDiscItem>
ReconfigurableRed::DoPeek()
{
    return GetInternalQueue(0)->Peek();
}

bool
ReconfigurableRed::CheckConfig()
{
    if (GetNInternalQueues() == 0)
    {
        AddInternalQueue(
            CreateObjectWithAttributes<DropTailQueue<QueueDiscItem>>(
                "MaxSize", QueueSizeValue(GetMaxSize())));
    }
    return GetNInternalQueues() == 1;
}

// ------------------------------------------------------------------ scenario

Ptr<QueueDisc> g_bottleneckQueue;
Ptr<ReconfigurableRed> g_reconfigurable;
uint32_t g_segmentSize = 1000;
double g_sampleInterval = 0.010;
double g_warmup = 8.0;
uint32_t g_samples = 1000;
uint32_t g_taken = 0;
uint32_t g_activeFlows = 0;
uint32_t g_cwndBytes = 0;
uint32_t g_ssThreshBytes = 0;
double g_controlInterval = 0.5;
double g_pMax = 0.1;
double g_minTh = 20.0;
bool g_controlOpen = true;
Ptr<PacketSink> g_taggedSink;

void
SetActiveFlows(uint32_t value)
{
    g_activeFlows = value;
}

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
    const int phase = (g_cwndBytes < g_ssThreshBytes) ? 0 : 1;
    const uint32_t drops =
        g_bottleneckQueue ? g_bottleneckQueue->GetStats().nTotalDroppedPackets : 0;
    // qAvg is appended because it is the variable the operator's state does
    // not contain and the AQM's drop decision does. It is RED's own internal
    // average, so a router has it by construction; recording it uses
    // information a predictive controller already holds rather than
    // privileged knowledge.
    const double qAvg = g_reconfigurable ? g_reconfigurable->GetQAvg() : -1.0;
    std::cout << "S," << std::fixed << std::setprecision(6)
              << Simulator::Now().GetSeconds() << ',' << cwndPkts << ','
              << queuePkts << ',' << phase << ',' << g_activeFlows << ','
              << drops << ',' << qAvg << '\n';
    ++g_taken;
    Simulator::Schedule(Seconds(g_sampleInterval), &Sample);
}

// One control instant: publish state, block for an action, apply it.
void
Control()
{
    if (!g_reconfigurable)
    {
        return;
    }
    const uint32_t queuePkts = g_bottleneckQueue->GetNPackets();
    const uint32_t drops = g_bottleneckQueue->GetStats().nTotalDroppedPackets;
    const uint64_t delivered =
        g_taggedSink ? g_taggedSink->GetTotalRx() / g_segmentSize : 0;
    const double cwndPkts = static_cast<double>(g_cwndBytes) / g_segmentSize;
    const int phase = (g_cwndBytes < g_ssThreshBytes) ? 0 : 1;

    if (g_controlOpen)
    {
        // n_active is published because a router knows how many sessions are
        // in its flow table. Withholding it made every controller condition on
        // the load the run started with, which is wrong for a third of a
        // cycling run and is not a limitation any deployment would have.
        std::cout << "CTRL," << std::fixed << std::setprecision(6)
                  << Simulator::Now().GetSeconds() << ','
                  << g_reconfigurable->GetQAvg() << ',' << queuePkts << ','
                  << cwndPkts << ',' << phase << ',' << drops << ','
                  << delivered << ',' << g_activeFlows << '\n'
                  << std::flush;
        std::string line;
        if (std::getline(std::cin, line) && !line.empty())
        {
            double pMax = g_pMax;
            double minTh = g_minTh;
            const size_t comma = line.find(',');
            if (comma != std::string::npos)
            {
                pMax = std::stod(line.substr(0, comma));
                minTh = std::stod(line.substr(comma + 1));
                g_pMax = pMax;
                g_minTh = minTh;
                g_reconfigurable->Reconfigure(pMax, minTh);
            }
        }
        else
        {
            // EOF: hold the last setting for the rest of the run. This is how
            // an uncontrolled reference goes through the same code path.
            g_controlOpen = false;
        }
    }
    Simulator::Schedule(Seconds(g_controlInterval), &Control);
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
    Ptr<TcpSocketBase> tcp = DynamicCast<TcpSocketBase>(socket);
    if (!tcp)
    {
        return;
    }
    tcp->TraceConnectWithoutContext("CongestionWindow", MakeCallback(&CwndTracer));
    tcp->TraceConnectWithoutContext("SlowStartThreshold", MakeCallback(&SsThreshTracer));
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
    uint32_t surgeFlows = 0;
    double surgeStartS = -1.0;
    double surgeStopS = -1.0;
    bool useStockRed = false;
    int64_t redStream = 7;

    CommandLine cmd(__FILE__);
    cmd.AddValue("capacityPps", "bottleneck capacity in packets per second", capacityPps);
    cmd.AddValue("rttPropS", "propagation round-trip time in seconds", rttPropS);
    cmd.AddValue("nFlows", "number of long-lived TCP flows", nFlows);
    cmd.AddValue("pErr", "independent packet error rate on the bottleneck", pErr);
    cmd.AddValue("pMax", "initial RED maximum drop probability", pMax);
    cmd.AddValue("minTh", "initial RED minimum queue threshold in packets", minTh);
    cmd.AddValue("maxThFrac", "RED maximum threshold as a fraction of the buffer",
                 maxThFrac);
    cmd.AddValue("bufferPkts", "bottleneck buffer size in packets", bufferPkts);
    cmd.AddValue("qw", "RED averaging weight", qw);
    cmd.AddValue("warmup", "seconds discarded before the first sample", g_warmup);
    cmd.AddValue("samples", "number of sample intervals after the first", g_samples);
    cmd.AddValue("sampleInterval", "seconds between samples", g_sampleInterval);
    cmd.AddValue("segmentSize", "TCP segment size in bytes", g_segmentSize);
    cmd.AddValue("runSeed", "ns-3 run number", runSeed);
    cmd.AddValue("sndBufPkts", "sender socket buffer in packets", sndBufPkts);
    cmd.AddValue("controlIntervalS", "seconds between control instants",
                 g_controlInterval);
    cmd.AddValue("surgeFlows", "extra sessions added at the load step", surgeFlows);
    cmd.AddValue("surgeStartS",
                 "seconds after the first sample when the extra sessions start; "
                 "negative disables the step",
                 surgeStartS);
    cmd.AddValue("surgeStopS",
                 "seconds after the first sample when they stop again; negative "
                 "leaves them running. A load that rises and falls is what makes "
                 "the best RED setting move: with a one-way step the controllers "
                 "converge on a single setting and agree on it 97% of the time, "
                 "so the loop cannot separate them however good their forecasts",
                 surgeStopS);
    cmd.AddValue("useStockRed",
                 "use ns-3's own RedQueueDisc, which cannot be reconfigured; "
                 "for the equivalence check only",
                 useStockRed);
    cmd.AddValue("redStream", "RNG stream for the queue disc", redStream);
    cmd.Parse(argc, argv);

    g_pMax = pMax;
    g_minTh = minTh;

    RngSeedManager::SetSeed(20260826);
    RngSeedManager::SetRun(runSeed);

    const uint64_t bottleneckBps =
        static_cast<uint64_t>(capacityPps * g_segmentSize * 8.0);
    const double oneWayEdge = rttPropS / 4.0;
    const double oneWayCore = rttPropS / 4.0;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName("ns3::TcpNewReno")));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::SndBufSize",
                       UintegerValue(sndBufPkts * g_segmentSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 22));

    const bool stepEnabled = surgeFlows > 0 && surgeStartS >= 0.0;
    const uint32_t totalFlows = nFlows + (stepEnabled ? surgeFlows : 0);

    NodeContainer senders;
    senders.Create(totalFlows);
    NodeContainer routers;
    routers.Create(2);
    NodeContainer receivers;
    receivers.Create(totalFlows);

    PointToPointHelper edge;
    edge.SetDeviceAttribute("DataRate", DataRateValue(DataRate(bottleneckBps * 10)));
    edge.SetChannelAttribute("Delay", TimeValue(Seconds(oneWayEdge)));
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

    const QueueSize bufferSize(QueueSizeUnit::PACKETS,
                               static_cast<uint32_t>(bufferPkts));
    TrafficControlHelper tch;
    if (useStockRed)
    {
        tch.SetRootQueueDisc("ns3::RedQueueDisc",
                             "MinTh", DoubleValue(minTh),
                             "MaxTh", DoubleValue(maxThFrac * bufferPkts),
                             "LInterm", DoubleValue(1.0 / pMax),
                             "QW", DoubleValue(qw),
                             "MaxSize", QueueSizeValue(bufferSize),
                             "UseEcn", BooleanValue(false),
                             "Gentle", BooleanValue(true),
                             "UseHardDrop", BooleanValue(false),
                             "MeanPktSize", UintegerValue(g_segmentSize));
    }
    else
    {
        tch.SetRootQueueDisc("ReconfigurableRed",
                             "MinTh", DoubleValue(minTh),
                             "MaxTh", DoubleValue(maxThFrac * bufferPkts),
                             "LInterm", DoubleValue(1.0 / pMax),
                             "QW", DoubleValue(qw),
                             "MaxSize", QueueSizeValue(bufferSize),
                             "MeanPktSize", UintegerValue(g_segmentSize));
    }
    QueueDiscContainer coreQueues = tch.Install(coreDevices);
    g_bottleneckQueue = coreQueues.Get(0);
    // A fixed stream on whichever disc is installed. Without this the two
    // implementations would draw from different substreams and could not be
    // compared trace against trace even if the algorithm were identical.
    if (useStockRed)
    {
        DynamicCast<RedQueueDisc>(g_bottleneckQueue)->AssignStreams(redStream);
    }
    else
    {
        g_reconfigurable = DynamicCast<ReconfigurableRed>(g_bottleneckQueue);
        g_reconfigurable->AssignStreams(redStream);
    }

    TrafficControlHelper fifo;
    fifo.SetRootQueueDisc("ns3::FifoQueueDisc",
                          "MaxSize", QueueSizeValue(QueueSize("1000p")));

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.255.0");
    address.Assign(coreDevices);

    ApplicationContainer sinks;
    ApplicationContainer sources;
    const uint16_t port = 50000;

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
        ApplicationContainer installed = sink.Install(receivers.Get(i));
        sinks.Add(installed);
        if (i == 0)
        {
            g_taggedSink = DynamicCast<PacketSink>(installed.Get(0));
        }

        BulkSendHelper source("ns3::TcpSocketFactory",
                              InetSocketAddress(rightInterfaces.GetAddress(1), port));
        source.SetAttribute("MaxBytes", UintegerValue(0));
        sources.Add(source.Install(senders.Get(i)));
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    sinks.Start(Seconds(0.0));
    for (uint32_t i = 0; i < nFlows; ++i)
    {
        sources.Get(i)->SetStartTime(Seconds(0.01 * i));
    }
    g_activeFlows = nFlows;

    if (stepEnabled)
    {
        const double surgeStart = g_warmup + surgeStartS;
        for (uint32_t i = nFlows; i < totalFlows; ++i)
        {
            sources.Get(i)->SetStartTime(Seconds(surgeStart + 0.01 * (i - nFlows)));
            if (surgeStopS >= 0.0)
            {
                sources.Get(i)->SetStopTime(Seconds(g_warmup + surgeStopS));
            }
        }
        Simulator::Schedule(Seconds(surgeStart), &SetActiveFlows, totalFlows);
        if (surgeStopS >= 0.0)
        {
            Simulator::Schedule(Seconds(g_warmup + surgeStopS), &SetActiveFlows,
                                nFlows);
        }
    }

    Simulator::Schedule(Seconds(0.05), &ConnectTaggedTraces, sources.Get(0));
    Simulator::Schedule(Seconds(g_warmup), &Sample);
    if (!useStockRed)
    {
        Simulator::Schedule(Seconds(g_warmup), &Control);
    }

    const double stop = g_warmup + (g_samples + 2) * g_sampleInterval;
    Simulator::Stop(Seconds(stop));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
