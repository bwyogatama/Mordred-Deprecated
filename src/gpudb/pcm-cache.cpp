/*
Copyright (c) 2009-2020, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// written by Roman Dementiev,
//            Thomas Willhalm,
//            Patrick Ungerer


/*!     \file pcm.cpp
\brief Example of using CPU counters: implements a simple performance counter monitoring utility
*/
#include <iostream>
#ifdef _MSC_VER
#include <windows.h>
#include "windows/windriver.h"
#else
#include <unistd.h>
#include <signal.h>   // for atexit()
#include <sys/time.h> // for gettimeofday()
#endif
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <sstream>
#include <assert.h>
#include <bitset>
#include "cpucounters.h"
// #include "pcm-pcie.h"
#include "utils.h"

#define SIZE (10000000)
#define PCM_DELAY_DEFAULT 1.0 // in seconds
#define PCM_DELAY_MIN 0.015 // 15 milliseconds is practical on most modern CPUs
#define MAX_CORES 4096

using namespace std;
using namespace pcm;

constexpr uint32 max_sockets = 256;
uint32 max_imc_channels = ServerUncoreCounterState::maxChannels;

template <class IntType>
double float_format(IntType n)
{
    return double(n) / 1e6;
}

std::string temp_format(int32 t)
{
    char buffer[1024];
    if (t == PCM_INVALID_THERMAL_HEADROOM)
        return "N/A";

    snprintf(buffer, 1024, "%2d", t);
    return buffer;
}

std::string l3cache_occ_format(uint64 o)
{
    char buffer[1024];
    if (o == PCM_INVALID_QOS_MONITORING_DATA)
        return "N/A";

    snprintf(buffer, 1024, "%6u", (uint32) o);
    return buffer;
}

template <class UncoreStateType>
double getAverageUncoreFrequencyGhz(const UncoreStateType& before, const UncoreStateType& after) // in GHz
{
    return getAverageUncoreFrequency(before, after) / 1e9;
}


// Static global variables defined for monitoring memory bandwidth.
static PCM *m = nullptr;

static ServerUncoreCounterState *BeforeUncoreState = nullptr;
static ServerUncoreCounterState *AfterUncoreState = nullptr;

static CoreCounterState *BeforeCoreState = nullptr;
static CoreCounterState *AfterCoreState = nullptr;

static SocketCounterState *BeforeSocketState = nullptr;
static SocketCounterState *AfterSocketState = nullptr;

static SystemCounterState *BeforeSystemState = nullptr;
static SystemCounterState *AfterSystemState = nullptr;

// static IPlatform *platform = nullptr;

static uint64_t BeforeTime = 0UL, AfterTime = 0UL;

void InitMonitor() {
    m = PCM::getInstance();
    fprintf(stderr, "After getting PCM instance\n");
    m->disableJKTWorkaround();
    // program() creates common semaphore for the singleton, so ideally to be called before any other references to PCM
    const PCM::ErrorCode status = m->program(PCM::DEFAULT_EVENTS, nullptr, false, -1);

    switch (status)
    {
    case PCM::Success:
        break;
    case PCM::MSRAccessDenied:
        cerr << "Access to Processor Counter Monitor has denied (no MSR or PCI CFG space access).\n";
        exit(EXIT_FAILURE);
    case PCM::PMUBusy:
        cerr << "Access to Processor Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU.\n";
        cerr << "Alternatively you can try running PCM with option -r to reset PMU.\n";
        exit(EXIT_FAILURE);
    default:
        cerr << "Access to Processor Counter Monitor has denied (Unknown error).\n";
        exit(EXIT_FAILURE);
    }
    
    if (!m->hasPCICFGUncore())
    {
        cerr << "Unsupported processor model (" << m->getCPUModel() << ").\n";
        if (m->memoryTrafficMetricsAvailable())
            cerr << "For processor-level memory bandwidth statistics please use 'pcm' utility\n";
        exit(EXIT_FAILURE);
    }

    if(m->getNumSockets() > max_sockets)
    {
        cerr << "Only systems with up to "<<max_sockets<<" sockets are supported! Program aborted" << endl;
        exit(EXIT_FAILURE);
    }

    // double delay = -1.0;
    // platform = IPlatform::getPlatform(m, false, true, true, (uint)delay);

    return;
}

// IPlatform *IPlatform::getPlatform(PCM *m, bool csv, bool print_bandwidth, bool print_additional_info, uint32 delay)
// {
//     switch (m->getCPUModel()) {
//         case PCM::ICX:
//         case PCM::SNOWRIDGE:
//             return new WhitleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
//         case PCM::SKX:
//             return new PurleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
//         case PCM::BDX_DE:
//         case PCM::BDX:
//         case PCM::KNL:
//         case PCM::HASWELLX:
//             return new GrantleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
//         case PCM::IVYTOWN:
//         case PCM::JAKETOWN:
//             return new BromolowPlatform(m, csv, print_bandwidth, print_additional_info, delay);
//         default:
//           return NULL;
//     }
// }



void StartMonitor() {
    // Allocate memory and store them using the statuc global variable.
    // If the end routine is not called then we have a memory leak
    BeforeUncoreState = new ServerUncoreCounterState[m->getNumSockets()];
    AfterUncoreState = new ServerUncoreCounterState[m->getNumSockets()];

    BeforeCoreState = new CoreCounterState[m->getNumCores()];
    AfterCoreState = new CoreCounterState[m->getNumCores()];

    BeforeSocketState = new SocketCounterState[m->getNumSockets()];
    AfterSocketState = new SocketCounterState[m->getNumSockets()];

    BeforeSystemState = new SystemCounterState;
    AfterSystemState = new SystemCounterState;

    for(uint32 i=0; i<m->getNumSockets(); ++i) {
        BeforeUncoreState[i] = m->getServerUncoreCounterState(i); 
    }

    for(uint32 i=0; i<m->getNumCores(); ++i) {
        BeforeCoreState[i] = m->getCoreCounterState(i); 
    }

    for(uint32 i=0; i<m->getNumSockets(); ++i) {
        BeforeSocketState[i] = m->getSocketCounterState(i); 
    }

    *BeforeSystemState = m->getSystemCounterState();

    BeforeTime = m->getTickCount();

    return;
}

void EndMonitor(long long& read, long long& write) {
    AfterTime = m->getTickCount();

    for(uint32 i=0; i<m->getNumSockets(); ++i) {
        AfterUncoreState[i] = m->getServerUncoreCounterState(i); 
    }

    for(uint32 i=0; i<m->getNumCores(); ++i) {
        AfterCoreState[i] = m->getCoreCounterState(i); 
    }

    for(uint32 i=0; i<m->getNumSockets(); ++i) {
        AfterSocketState[i] = m->getSocketCounterState(i); 
    }

    *AfterSystemState = m->getSystemCounterState();

    // int tot_misses = 0;
    // for(uint32 i=0; i<m->getNumCores(); ++i) {
    //     int misses = getL3CacheMisses(BeforeCoreState[i], AfterCoreState[i]);
    //     tot_misses += misses;
    // }
    // cout << tot_misses << endl;

    // int tot_misses = 0;
    // for(uint32 i=0; i<m->getNumSockets(); ++i) {
    //     int misses = getL3CacheMisses(BeforeSocketState[i], AfterSocketState[i]);
    //     cout << unit_format(getL3CacheMisses(BeforeSocketState[i], AfterSocketState[i])) << endl;
    //     tot_misses += misses;
    // }
    // cout << tot_misses << endl;

    // cout << unit_format(getL2CacheMisses(*BeforeSystemState, *AfterSystemState)) << endl;

    // for(uint32 i=0; i<m->getNumSockets(); ++i) {
    //     int misses = getL3CacheMisses(BeforeSocketState[i], AfterSocketState[i]);
    //     cout << unit_format(getL3CacheMisses(BeforeSocketState[i], AfterSocketState[i])) << endl;
    //     tot_misses += misses;
    // }
    
    // calculate_bandwidth(m, BeforeUncoreState, AfterUncoreState, AfterTime-BeforeTime, 2);

    // max_imc_channels = (pcm::uint32)m->getMCChannelsPerSocket();

    long long reads = 0, writes = 0;
    for(uint32 skt = 0; skt < m->getNumSockets(); ++skt) {
        for (uint32 channel = 0; channel < max_imc_channels; ++channel) {
            reads += getMCCounter(channel, ServerPCICFGUncore::EventPosition::READ, BeforeUncoreState[skt], AfterUncoreState[skt]);
            writes += getMCCounter(channel, ServerPCICFGUncore::EventPosition::WRITE, BeforeUncoreState[skt], AfterUncoreState[skt]);
        }
    }

    cout << reads * 64 << " " << writes * 64 << endl;

    read += reads * 64;
    write += writes * 64;

    delete[] BeforeCoreState;
    delete[] AfterCoreState;
    delete[] BeforeUncoreState;
    delete[] AfterUncoreState;
    delete[] BeforeSocketState;
    delete[] AfterSocketState;
    delete BeforeSystemState;
    delete AfterSystemState;

    return;
}