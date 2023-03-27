#include "onnx.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <iomanip>
#include <fstream>
#include <string>

static float get_node_attr_f(const onnx::NodeProto& node, const char* key, float def=0.f)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.f();
        }
    }
    return def;
}

static float get_node_attr_i(const onnx::NodeProto& node, const char* key, float def=0.f)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return std::max(std::min(attr.i(), (::google::protobuf::int64)INT_MAX), (::google::protobuf::int64)INT_MIN);
        }
    }
    return def;
}

static onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.t();
        }
    }
    return onnx::TensorProto();
}

static int get_tensor_proto_data_size(const onnx::TensorProto& tp)
{
    const std::string& raw_data = tp.raw_data();
    int size = (int)raw_data.size() / 4;
    return size;
}

static bool read_onnx_model(const char* filepath, onnx::ModelProto& message)
{
    std::ifstream ifs(filepath, std::ifstream::in | std::ifstream::binary);
    
    if (!ifs.is_open())
    {
        fprintf(stderr, "Open failed %s\n", filepath);
        return false;
    }
    
    if (!message.ParseFromIstream(&ifs))
    {
        fprintf(stderr, "Failed to parse onnx model.%s\n", filepath);
        return false;
    }

    return true;
}


int main(int argc, char** argv)
{
    if (!(argc == 2 || argc ==4))
    {
        fprintf(stderr, "Usage: %s [onnxpb] [param] [bin]\n", argv[0]);
        return -1;
    }

    const char* onnxpb = argv[1];
    const char* tinyinfer_prorotxt = argc == 2 ? "tinyinfer.param" : argv[2];
    const char* tinyinfer_modelbin = argc == 2 ? "tinyinfer.bin" : argv[3];

    onnx::ModelProto model;
    bool s1 = read_onnx_model(onnxpb, model);
    if (!s1)
    {
        fprintf(stderr, "read onnx model failed\n");
        return -1;
    }

    std::ofstream pofs(tinyinfer_prorotxt, std::fstream::out);
    std::ofstream bofs(tinyinfer_modelbin, std::fstream::out | std::fstream::binary);

    pofs << "202303" << std::endl;

    const onnx::GraphProto& graph = model.graph();
    onnx::GraphProto* mutable_graph = model.mutable_graph();
    int node_num = graph.node_size();

    std::map<std::string, int> node_reference_cnt;
    std::map<std::string, onnx::TensorProto> weights;

    for (int i = 0; i < graph.initializer_size(); i++)
    {
        const onnx::TensorProto& initializer = graph.initializer(i);
        fprintf(stderr, "weight = %s %d\n", initializer.name().c_str(), initializer.data_type());
        weights[initializer.name()] = initializer;
    }

    {
        // topological order
        std::set<std::string> producers;
        for (int i = 0; i < graph.input_size(); i++)
        {
            const std::string& input_name = graph.input(i).name();
            producers.insert(input_name);
        }

        for (int i = 0; i < node_num;)
        {
            onnx::NodeProto* node = mutable_graph->mutable_node(i);
            
            // swapnode: the input of this node comes form its subsequent node
            bool swapnode = false; 
            std::string missing_input_name;
            for (int j = 0; j < (int)node->input_size(); j++)
            {
                const std::string& input_name = node->input(j);
                if (input_name.empty())
                    continue;
                
                if (producers.find(input_name) == producers.end() && weights.find(input_name) == weights.end())
                {
                    swapnode = true;
                    missing_input_name = input_name;
                    break;
                }
            }

            // if the node is not swapnode, insert it into producers, then judge the next node
            if (!swapnode)
            {
                for (int j = 0; j < (int)node->output_size(); j++)
                {
                    const std::string& output_name = node->output(j);
                    if (output_name.empty())
                        continue;
                    
                    producers.insert(output_name);
                }
                
                i++;
                continue;
            }

            // find node that produces missing_input_name
            int q = i + 1;
            for (; q < node_num; q++)
            {
                onnx::NodeProto* nodeq = mutable_graph->mutable_node(q);
                bool found = false;
                for (int k = 0; k < (int)nodeq->output_size(); k++)
                {
                    const std::string& output_name = nodeq->output(k);
                    if (output_name == missing_input_name)
                    {
                        found = true;
                        break;
                    }
                }
                
                if (found)
                    break;
            }

            // fail to find node that produces missing_input_name 
            if (q == node_num)
            {
                fprintf(stderr, "cannot find node produces %s but node %d requires it\n", missing_input_name.c_str(), i);
                return -1;
            }

            // succeed, swap the pair of nodes
            fprintf(stderr, "swap node %d %d", i, q);
            onnx::NodeProto* nodeq = mutable_graph->mutable_node(q);
            onnx::NodeProto tmp = *node;
            *node = *nodeq;
            *nodeq = tmp;
        }
    }

    // collect blobs
    std::set<std::string> blob_names;
    for (int i = 0; i < node_num; i++)
    {
        const onnx::NodeProto& node = graph.node(i);
        const std::string op = node.op_type();
        std::string name = node.name();

        if(name.empty())
        {
            name = node.output(0);
        }

        // treat constant node as weight or binaryop_weights
        if(op == "Constant")
        {
            onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
            weights[node.output(0)] = tensor;
        }

        for (int j = 0; j < (int)node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            blob_names.insert(input_name);
            if (node_reference_cnt.find(input_name) == node_reference_cnt.end())
            {
                node_reference_cnt[input_name] = 1;
            }
            else
            {
                node_reference_cnt[input_name] += 1;
            }
        }

        if (op == "Dropout")
        {
            const std::string& output_name = node.output(0);
            blob_names.insert(output_name);
            node_reference_cnt[output_name] = 0;
            continue;
        }

        for (int j = 0; j < (int)node.output_size(); j++)
        {
            const std::string& output_name = node.output(j);
            blob_names.insert(output_name);
            node_reference_cnt[output_name] = 0;
        }
    }

    int input_node_cnt = 0;
    for (int i = 0; i < graph.input_size(); i++)
    {
        const std::string& input_name = graph.input(i).name();
        if (weights.find(input_name) != weights.end())
            continue;
        blob_names.insert(input_name);
        input_node_cnt++;
    }

    // fprintf(stderr, "node num: %d blob num: %ld\n", node_num, blob_names.size());
    int reduced_node_cnt = 0;
    // fuse operations

    // reduce common const weight node_reference
    for (int i = 0; i < node_num; i++)
    {
        const onnx::NodeProto& node = graph.node(i);
        const std::string& op = node.op_type();
        
        if (op == "BatchNormalization")
        {
            node_reference_cnt[node.input(1)] -= 1;
            node_reference_cnt[node.input(2)] -= 1;
            node_reference_cnt[node.input(3)] -= 1;
            node_reference_cnt[node.input(4)] -= 1;
        }
        else if (op == "Conv")
        {
            node_reference_cnt[node.input(1)] -= 1;
            if (node.input_size() == 3)
            {
                node_reference_cnt[node.input(2)] -= 1;
            }
        }
        else if (op == "Gemm")
        {
            float alpha = get_node_attr_f(node, "alpha", 1.f);
            float beta = get_node_attr_f(node, "beta", 1.f);
            int transA = get_node_attr_i(node, "transA", 0);
            int transB = get_node_attr_i(node, "transB", 0);

            if (alpha == 1.f && beta == 1.f && transA == 0 && transB == 1)
            {
                // InnerProduct-like A * B + C
                node_reference_cnt[node.input(1)] -= 1;
                node_reference_cnt[node.input(2)] -= 1;
            }
        }
    }

    int zero_inference_weight_node_cnt = 0;
    for (std::map<std::string, onnx::TensorProto>::iterator it = weights.begin(); it != weights.end(); it++)
    {
        const std::string& input_name = it->first;

        // there may be some weight nodes in initializer but none of the graph node use them
        // add them to blob_names so we could get proper blob count later
        blob_names.insert(input_name);

        int refcount = node_reference_cnt[input_name];
        if (refcount == 0)
            zero_inference_weight_node_cnt++;
    }

    // we always treat constant node as weight or binaryop_weights
    // do not count it twice for layer_count
    int constant_node_count_moved_to_weight = 0;
    for (int i = 0; i < node_num; i++)
    {
        const onnx::NodeProto& node = graph.node(i);

        const std::string& op = node.op_type();

        if (op == "Constant")
        {
            constant_node_count_moved_to_weight++;
        }
    }

    blob_names.erase("");
    node_reference_cnt.erase("");

    int split_layer_count = 0;
    int splittinyinfer_blob_cnt = 0;
    std::map<std::string, int> split_node_reference;
    for (std::map<std::string, int>::iterator it = node_reference_cnt.begin(); it != node_reference_cnt.end(); it++)
    {
        if (it->second > 1)
        {
            split_layer_count++;
            splittinyinfer_blob_cnt += it->second;
            split_node_reference[it->first] = it->second;
        }
    }

    int blob_num1 = node_num + input_node_cnt + splittinyinfer_blob_cnt - reduced_node_cnt \
                    - constant_node_count_moved_to_weight + weights.size() - zero_inference_weight_node_cnt;
    int blob_num2 = blob_names.size() - zero_inference_weight_node_cnt + splittinyinfer_blob_cnt;
    
    pofs << blob_num1 << " " << blob_num2 << std::endl;

    int internal_split = 0;

    // input information line
    for (int i = 0; i < graph.input_size(); i++)
    {
        const std::string& input_name = graph.input(i).name();

        if(weights.find(input_name) != weights.end())
            continue;

        pofs << std::left << std::setw(16) << "Input" << " " << std::setw(24) << input_name << " ";
        pofs << "0 1 " << input_name << std::endl;

        int refcount = node_reference_cnt[input_name];
        if (refcount <= 1)
            continue;
        
        std::string splitname = "split_tinyinfer_input" + std::to_string(i);
        pofs << std::left << std::setw(16) << "Split" << " " << std::setw(24) << splitname << " ";
        pofs << 1 << " " << refcount << " " << input_name;

        for (int j = 0; j < refcount; j++)
        {
            std::string split_name = input_name + "_split_" + std::to_string(j);
            pofs << " " << split_name;
        }
        pofs << std::endl;
    }

    // MemoryData information line
    for (std::map<std::string, onnx::TensorProto>::iterator weight_it = weights.begin(); weight_it != weights.end(); weight_it++)
    {
        const std::string& input_name = weight_it->first;

        int refcount = node_reference_cnt[input_name];
        if (refcount == 0)
            continue;

        pofs << std::left << std::setw(16) << "MemoryData" << " " << std::setw(24) << input_name << " ";
        pofs << "0 1 " << input_name;
        
        const onnx::TensorProto& M = weights[input_name];

        if (M.dims_size() == 0)
        {
            pofs << " 0=" << get_tensor_proto_data_size(M);
        }
        else if (M.dims_size() == 1)
        {
            pofs << " 0=" << (int)M.dims(0);
        }
        else if (M.dims_size() == 2)
        {
            pofs << " 0=" << (int)M.dims(1);
            if (M.dims(0) != 1)
                pofs << " 1=" << (int)M.dims(0);
        }
        else if (M.dims_size() == 3)
        {
            pofs << " 0=" << (int)M.dims(2);
            pofs << " 1=" << (int)M.dims(1);
            if (M.dims(0) != 1)
                pofs << " 2=" << (int)M.dims(0);
        }
        else if (M.dims_size() == 4)
        {
            pofs << " 0=" << (int)M.dims(3);
            pofs << " 1=" << (int)M.dims(2);
            pofs << " 2=" << (int)M.dims(1);
        }
        pofs << std::endl;

        if (refcount <= 1)
            continue;

        std::string splitname = "split_tinyinfer_" + std::to_string(internal_split);
        pofs << std::left << std::setw(16) << "Split" << std::setw(24) << splitname;
        pofs << " 1" << " " << refcount;
        pofs << " " << input_name;

        for (int j = 0; j < refcount; j++)
        {
            std::string split_name = input_name + "_split_" + std::to_string(j);
            pofs << " " << split_name;
        }
        pofs << std::endl;

        internal_split++;
    }

    pofs.close();
    bofs.close();
}