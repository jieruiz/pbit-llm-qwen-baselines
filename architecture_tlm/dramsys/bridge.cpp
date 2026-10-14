// Persistent JSON-lines bridge. Actual nonblocking 4-phase TLM requests;
// no b_transport constant-delay shortcut and no offline timestamp replay.
#include <DRAMSys/DRAMSys.h>
#include <DRAMSys/common/MemoryManager.h>
#include <DRAMSys/configuration/memspec/MemSpec.h>
#include <DRAMSys/power/DRAMPowerAdapter.h>
#include <DRAMSys/statistics/JsonFormat.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>
using json=nlohmann::json;

class Initiator:public sc_core::sc_module {
public:
    tlm_utils::simple_initiator_socket<Initiator> socket{"socket"};
    SC_HAS_PROCESS(Initiator);
    Initiator(sc_core::sc_module_name name,std::ostream& output,
              std::uint64_t capacity,unsigned maxPending):
        sc_module(name),out(output),capacity(capacity),limit(maxPending),
        peq(this,&Initiator::phase),mm(false) {
        socket.register_nb_transport_bw(this,&Initiator::backward);
        SC_THREAD(run);
    }
    std::uint64_t bursts=0,readBursts=0,writeBursts=0;
    unsigned peakPending=0;
    double requestedEnd=0;
private:
    std::ostream& out;
    std::uint64_t capacity;
    unsigned limit,pending=0;
    bool requestBusy=false;
    sc_core::sc_event endRequest,completed;
    tlm_utils::peq_with_cb_and_phase<Initiator> peq;
    DRAMSys::MemoryManager mm;
    tlm::tlm_sync_enum backward(tlm::tlm_generic_payload& p,tlm::tlm_phase& phase,
                               sc_core::sc_time& delay) {
        peq.notify(p,phase,delay);
        return tlm::TLM_ACCEPTED;
    }
    void phase(tlm::tlm_generic_payload& p,const tlm::tlm_phase& phase) {
        if(phase==tlm::END_REQ) {
            requestBusy=false;endRequest.notify(sc_core::SC_ZERO_TIME);
        } else if(phase==tlm::BEGIN_RESP) {
            if(p.is_response_error()) SC_REPORT_FATAL("bridge","memory response error");
            --pending;
            tlm::tlm_phase finish=tlm::END_RESP;
            auto delay=sc_core::sc_time::from_value(1);
            socket->nb_transport_fw(p,finish,delay);
            p.release();completed.notify(sc_core::SC_ZERO_TIME);
        } else SC_REPORT_FATAL("bridge","unexpected backward phase");
    }
    void advance(double ns) {
        auto target=sc_core::sc_time(ns,sc_core::SC_NS);
        if(target>sc_core::sc_time_stamp()) wait(target-sc_core::sc_time_stamp());
    }
    void run() {
        out<<json({{"status","ready"},{"memory_bytes",capacity}}).dump()<<std::endl;
        std::string line;
        while(std::getline(std::cin,line)) {
            auto r=json::parse(line);
            if(r.at("op")=="finish") {
                requestedEnd=r.at("end_ns").get<double>();
                advance(requestedEnd);
                sc_core::sc_stop();return;
            }
            if(r.at("op")!="access") throw std::runtime_error("unknown operation");
            auto address=r.at("address").get<std::uint64_t>();
            auto length=r.at("bytes").get<std::uint64_t>();
            if(!length||address%64||length%64||address>=capacity||length>capacity-address)
                throw std::runtime_error("unaligned or out-of-bounds DMA");
            advance(r.at("ready_ns").get<double>());
            auto start=sc_core::sc_time_stamp();
            bool write=r.at("write").get<bool>();
            for(std::uint64_t offset=0;offset<length;offset+=64) {
                if(requestBusy) wait(endRequest);
                while(pending>=limit) wait(completed);
                auto* p=mm.allocate(64);p->acquire();
                p->set_address(address+offset);p->set_data_length(64);p->set_streaming_width(64);
                p->set_command(write?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
                p->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
                p->set_dmi_allowed(false);p->set_byte_enable_ptr(nullptr);
                p->set_byte_enable_length(0);
                requestBusy=true;++pending;peakPending=std::max(peakPending,pending);
                ++bursts;if(write)++writeBursts;else++readBursts;
                tlm::tlm_phase phase=tlm::BEGIN_REQ;auto delay=sc_core::SC_ZERO_TIME;
                auto status=socket->nb_transport_fw(*p,phase,delay);
                if(status!=tlm::TLM_ACCEPTED) SC_REPORT_FATAL("bridge","unexpected forward status");
                // 1 GHz DMA issue front end, distinct from the 100 MHz SC array.
                wait(sc_core::sc_time(1,sc_core::SC_NS));
            }
            while(pending) wait(completed);
            out<<json({{"id",r.at("id")},{"done_ns",sc_core::sc_time_stamp().to_seconds()*1e9},
                       {"start_ns",start.to_seconds()*1e9},{"burst_count",length/64}}).dump()<<std::endl;
        }
        throw std::runtime_error("EOF without finish");
    }
};

void powerAt(sc_core::sc_object* obj,std::uint64_t cycle,json& energy) {
    if(auto* adapter=dynamic_cast<DRAMSys::DRAMPowerAdapter*>(obj)) {
        // Official adapter's stats use last command time. Evaluate the official
        // DRAMPower API at the full system horizon, including the compute tail.
        auto& variant=const_cast<DRAMSys::DRAMPowerVariant&>(adapter->getDRAMPowerVariant());
        std::visit([&](auto& model) {
            auto stats=model.getWindowStats(cycle);
            double core=model.calcCoreEnergy(stats).total();
            double interface=model.calcInterfaceEnergy(stats).total();
            energy[obj->basename()]={{"core_J",core},{"interface_J",interface},{"total_J",core+interface}};
        },variant);
    }
    for(auto* child:obj->get_child_objects()) powerAt(child,cycle,energy);
}

int sc_main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("usage: bridge configuration.json max_pending");
        sc_core::sc_set_time_resolution(1,sc_core::SC_PS);
        std::ostream output(std::cout.rdbuf());
        std::cout.rdbuf(std::cerr.rdbuf()); // upstream logs never pollute JSON protocol
        auto config=DRAMSys::Config::from_path(argv[1]);
        DRAMSys::DRAMSys dram("dram",config);
        auto limit=std::stoul(argv[2]);
        if(!limit) throw std::runtime_error("max_pending must be positive");
        Initiator initiator("dma",output,dram.memorySize(),limit);
        initiator.socket.bind(dram.tSocket);
        sc_core::sc_start();
        json stats,energy;
        DRAMSys::Stats::JsonFormat::collectStats(&dram,stats);
        auto cycle=static_cast<std::uint64_t>(std::floor(sc_core::sc_time_stamp()/dram.getMemSpec().tCK));
        powerAt(&dram,cycle,energy);
        output<<json({{"status","finished"},{"stats",stats},{"power_at_system_end",energy},
            {"end_ns",sc_core::sc_time_stamp().to_seconds()*1e9},{"dram_end_cycle",cycle},
            {"burst_count",initiator.bursts},{"read_bursts",initiator.readBursts},
            {"write_bursts",initiator.writeBursts},{"peak_pending",initiator.peakPending}}).dump()<<std::endl;
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
