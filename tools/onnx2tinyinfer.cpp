#include "onnx.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <float.h>

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

static std::vector<int> get_node_attr_ai(const onnx::NodeProto& node, const char* key)
{
    std::vector<int> v;
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == "key")
        {
            v.resize(attr.ints_size());
            for (int j = 0; j < attr.ints_size(); j++)
            {
                v[j] = std::max(std::min(attr.ints(j), (::google::protobuf::int64)INT_MAX), (::google::protobuf::int64)INT_MIN);
            }
            break;
        }
    }

    return v;
}

static std::string get_node_attr_s(const onnx::NodeProto& node, const char* key, const std::string& def = std::string())
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.s();
        }
    }

    return def;
}

static float get_node_attr_from_input_f(const onnx::TensorProto& tp)
{
    float v = 0.f;

    // float
    if (tp.data_type() == 1)
    {
        const float* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const float*)tp.raw_data().data();
        }
        else
        {
            shape_data = tp.float_data().data();
        }
        v = shape_data[0];
    }
    // double
    else if (tp.data_type() == 11)
    {
        const double* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const double*)tp.raw_data().data();
        }
        else
        {
            shape_data = tp.double_data().data();
        }
        v = shape_data[0];
    }
    // int64
    else if (tp.data_type() == 7)
    {
        const int64_t* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const int64_t*)tp.raw_data().data();
        }
        else
        {
            shape_data = tp.int64_data().data();
        }
        v = std::max(std::min(shape_data[0], (::google::protobuf::int64)INT_MAX), (::google::protobuf::int64)INT_MIN);
    }
    // int32
    else if (tp.data_type() == 6)
    {
        const int32_t* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const int32_t*)tp.raw_data().data();
        }
        else
        {
            shape_data = tp.int32_data().data();
        }
        v = shape_data[0];
    }
    else
    {
        fprintf(stderr, "Unknown data type %d\n", tp.data_type());
        abort();
    }

    return v;
}

static std::vector<int> get_node_attr_from_input_ai(const onnx::TensorProto& tp)
{
    int size = 0;

    std::vector<int> v;

    // int64
    if (tp.data_type() == 7)
    {
        const int64_t* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const int64_t*)tp.raw_data().data();
            size = (int)(tp.raw_data().size() / 8);
        }
        else
        {
            shape_data = tp.int64_data().data();
            size = tp.int64_data_size();
        }
        for (int j = 0; j < size; j++)
        {
            int vi = std::max(std::min(shape_data[j], (::google::protobuf::int64)INT_MAX), (::google::protobuf::int64)INT_MIN);
            v.push_back(vi);
        }
    }
    // int32
    else if (tp.data_type() == 6)
    {
        const int32_t* shape_data = 0;
        if (tp.raw_data().size())
        {
            shape_data = (const int32_t*)tp.raw_data().data();
            size = (int)(tp.raw_data().size() / 4);
        }
        else
        {
            shape_data = tp.int32_data().data();
            size = tp.int32_data_size();
        }
        for (int j = 0; j < size; j++)
        {
            v.push_back(shape_data[j]);
        }
    }
    else
    {
        fprintf(stderr, "Unknown data type %d\n", tp.data_type());
    }

    return v;
}

static int get_tensor_proto_data_size(const onnx::TensorProto& tp)
{
    if (tp.raw_data().size() > 0)
    {
        const std::string& raw_data = tp.raw_data();
        int size = (int)raw_data.size() / 4;
        return size;
    }
    else if (tp.data_type() == 1)
    {
        return tp.float_data_size();
    }
    return 0;
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

static void ofstream_tensor_proto_data(const onnx::TensorProto& tp, std::ofstream& ofs)
{
    int size = get_tensor_proto_data_size(tp);

    if (tp.raw_data().size() > 0)
    {
        const std::string& raw_data = tp.raw_data();
        ofs.write(raw_data.data(), size);
    }else if (tp.data_type() == 1)
    {
        ofs.write((const char*)tp.float_data().data(), size);
    }
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
        // fprintf(stderr, "weight = %s %d\n", initializer.name().c_str(), initializer.data_type());
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

        ofstream_tensor_proto_data(M, bofs);

        if (refcount <= 1)
            continue;

        std::string splitname = "split_tinyinfer_" + std::to_string(internal_split);
        pofs << std::left << std::setw(16) << "Split" << " " << std::setw(24) << splitname;
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

    // Node information line
    // [tinyinfer_op_name] [node_name] [input_num] [out_num] [input names] [output names] [attributes]
    for (int i = 0; i < node_num; i++)
    {
        const onnx::NodeProto& node = graph.node(i);
        const std::string& op = node.op_type();

        if (op == "noop_reduced")
            continue;
        
        std::string node_name = node.name();
        if (node_name.empty())
        {
            node_name = node.output(0);
        }

        int input_size = node.input_size();
        int output_size = node.output_size();

        for (int j = 0; j < (int)node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            // check weight
            if (weights.find(input_name) != weights.end() && node_reference_cnt[input_name] == 0)
                input_size--;

            if (input_name.empty())
                input_size--;
        }

        // [tinyinfer_op_name] [attributes]
        std::string tinyinfer_op_name;
        std::string attributes = "";
        if (op == "Abs")
        {
            tinyinfer_op_name = "UnaryOp";

            int op_type = 0;
            attributes = "0=" + std::to_string(op_type);
        }
        else if (op == "Acos")
        {
            tinyinfer_op_name = "UnaryOp";

            int op_type = 13;
            attributes = "0=" + std::to_string(op_type);
        }
        else if (op == "Add")
        {
            tinyinfer_op_name = "BinaryOp";

            int op_type = 0;
            attributes = "0=" + std::to_string(op_type);

            // int with_scalar = get_node_attr_i(node, "with_scalar", 0);
            // float b = get_node_attr_f(node, "b", 0.f);
            // if (with_scalar)
            // {
            //     attributes += " 1=" + std::to_string(with_scalar);
            //     attributes += " 2=" + std::to_string(b);
            // }
        }
        else if (op == "Asin")
        {
            tinyinfer_op_name = "UnaryOp";
            int op_type = 12;
            attributes = "0=" + std::to_string(op_type);
        }
        else if (op == "AveragePool" || op == "MaxPool")
        {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#AveragePool-11
            std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
            if (kernel_shape.size() == 1)
                tinyinfer_op_name = "Pooling1D";
            else
                tinyinfer_op_name = "Pooling";
            
            std::string auto_pad = get_node_attr_s(node, "auto_pad");
            int ceil_mode = get_node_attr_i(node, "ceil_mode", 0);
            std::vector<int> strides = get_node_attr_ai(node, "strides");
            std::vector<int> pads = get_node_attr_ai(node, "pads");

            // Op category
            int pool = op == "AveragePool" ? 1 : 0;
            attributes += "0=" + std::to_string(pool);

            // kernel_shape
            if (kernel_shape.size() == 1)
            {
                attributes += " 1=" + std::to_string(kernel_shape[0]);
            }
            else if (kernel_shape.size() == 2)
            {
                attributes += " 1=" + std::to_string(kernel_shape[1]);
                attributes += " 11=" + std::to_string(kernel_shape[0]);
            }

            // strides
            if (strides.size() == 1)
            {
                attributes += " 2=" + std::to_string(strides[0]);
            }
            else if (strides.size() == 2)
            {
                attributes += " 2=" + std::to_string(strides[1]);
                attributes += " 12=" + std::to_string(strides[0]);
            }

            // pads
            if (pads.size() == 1)
            {
                attributes += " 3=" + std::to_string(strides[0]);
            }
            else if (pads.size() == 2)
            {
                attributes += " 3=" + std::to_string(strides[1]);
                attributes += " 13=" + std::to_string(strides[0]);
            }
            else if (pads.size() == 4)
            {
                attributes += " 3=" + std::to_string(strides[1]);
                attributes += " 13=" + std::to_string(strides[0]);
                attributes += " 14=" + std::to_string(strides[3]);
                attributes += " 15=" + std::to_string(strides[2]);
            }

            // auto_pad
            int pad_mode = 1;
            if (auto_pad == "SAME_UPPER")
            {
                pad_mode = 2;
            }
            else if (auto_pad == "SAME_LOWER")
            {
                pad_mode = 3;
            }
            
            // ceil_mode
            if (ceil_mode == 1)
            {
                pad_mode = 0;
            }
            attributes += " 5=" + std::to_string(pad_mode);

            // count_include_pad
            if (op == "AveragePool")
            {
                int avgpool_count_include_pad = get_node_attr_i(node, "count_include_pad", 0);
                attributes += " 6=" + std::to_string(avgpool_count_include_pad);
            }
        }
        else if (op == "BatchNormalization")
        {
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
            tinyinfer_op_name = "BatchNorm";
            float epsilon = get_node_attr_f(node, "epsilon", 1e-5f);

            const onnx::TensorProto& scale = weights[node.input(1)];
            const onnx::TensorProto& B = weights[node.input(2)];
            const onnx::TensorProto& mean = weights[node.input(3)];
            const onnx::TensorProto& var = weights[node.input(4)];
            int channels = get_tensor_proto_data_size(scale);

            attributes += "0=" + std::to_string(channels);

            ofstream_tensor_proto_data(scale, bofs);
            ofstream_tensor_proto_data(mean, bofs);
            {
                const float* v = var.raw_data().size() ? (const float*)var.raw_data().data() : var.float_data().data();
                for (int j = 0; j < channels; j++)
                {
                    float ve = v[j] + epsilon;
                    bofs.write((const char*)&ve, sizeof(float));
                }
            }
        }
        else if (op == "Clip")
        {
            tinyinfer_op_name = "Clip";
            float min;
            float max;
            if (node.input_size() == 1)
            {
                min = get_node_attr_f(node, "min", -FLT_MAX);
                max = get_node_attr_f(node, "max", FLT_MAX);
            }
            else
            {
                min = weights.find(node.input(1)) != weights.end() ? get_node_attr_from_input_f(weights[node.input(1)]) : -FLT_MAX;
                max = weights.find(node.input(2)) != weights.end() ? get_node_attr_from_input_f(weights[node.input(2)]) : FLT_MAX;
            }
            attributes += "0=" + std::to_string(min);
            attributes += " 1=" + std::to_string(max);
        }
        else if (op == "Concat")
        {
            tinyinfer_op_name = "Constant";
            
            int axis = get_node_attr_i(node, "axis", 1);
            attributes += "0=" + std::to_string(axis > 0 ? axis - 1 : axis);
        }
        else if (op == "Constant")
        {
            continue;
        }
        else if (op == "Conv")
        {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Conv-11
            std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
            if (kernel_shape.size() == 1)
            {
                tinyinfer_op_name = "Convolution1D";
            }
            else
            {
                int group = get_node_attr_i(node, "group", 1);
                if (group > 1)
                {
                    tinyinfer_op_name = "ConvolutionDepthWise";
                }
                else
                {
                    tinyinfer_op_name = "Convolution";
                }
            }

            const onnx::TensorProto& W = weights[node.input(1)];

            int num_filter = W.dims(0);
            int has_bias = node.input_size() == 3 ? 1 : 0;

            std::string auto_pad = get_node_attr_s(node, "auto_pad");
            std::vector<int> dilations = get_node_attr_ai(node, "dilations");
            std::vector<int> strides = get_node_attr_ai(node, "strides");
            std::vector<int> pads = get_node_attr_ai(node, "pads");
            int group = get_node_attr_i(node, "group", 1);

            // filter number
            attributes += "0=" + std::to_string(num_filter);

            // kernel shape
            if (kernel_shape.size() == 1)
            {
                attributes += " 1=" + std::to_string(kernel_shape[0]);
            }
            else if (kernel_shape.size() == 2)
            {
                attributes += " 1=" + std::to_string(kernel_shape[1]);
                attributes += " 11=" + std::to_string(kernel_shape[0]);
            }
            
            // dilation size
            if (dilations.size() == 1)
            {
                attributes += " 2=" + std::to_string(dilations[0]);
            }
            else if (dilations.size() == 2)
            {
                attributes += " 2=" + std::to_string(dilations[1]);
                attributes += " 12=" + std::to_string(dilations[0]);
            }

            // stride size
            if (strides.size() == 1)
            {
                attributes += " 3=" + std::to_string(strides[0]);
            }
            else if (strides.size() == 2)
            {
                attributes += " 3=" + std::to_string(strides[1]);
                attributes += " 13=" + std::to_string(strides[0]);
            }

            // pads
            if (auto_pad == "SAME_UPPER")
            {
                attributes += " 4=-233";
            }
            else if (auto_pad == "SAME_LOWER")
            {
                attributes += " 4=-234";
            }
            else
            {
                if (pads.size() == 1)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                }
                else if (pads.size() == 2)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                    attributes += " 14=" + std::to_string(pads[0]);
                }
                else if (pads.size() == 4)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                    attributes += " 14=" + std::to_string(pads[0]);
                    attributes += " 15=" + std::to_string(pads[3]);
                    attributes += " 16=" + std::to_string(pads[2]);
                }
            }

            attributes += " 5=" + std::to_string(has_bias);
            attributes += " 6=" + std::to_string(get_tensor_proto_data_size(W));
            
            if (group > 1)
            {
                attributes += " 7=" + std::to_string(group);
            }

            ofstream_tensor_proto_data(W, bofs);
            if (has_bias)
            {
                const onnx::TensorProto& B = weights[node.input(2)];
                ofstream_tensor_proto_data(B, bofs);
            }
        }
        else if (op == "ConvTranspose")
        {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConvTranspose-11
            int group = get_node_attr_i(node, "group", 1);
            if (group > 1)
            {
                tinyinfer_op_name = "DeConvolutionDepthWise";
            }
            else
            {
                tinyinfer_op_name = "DeConvolution";
            }

            const onnx::TensorProto& W = weights[node.input(1)];

            int has_bias = node.input_size() == 3 ? 1 : 0;

            std::string auto_pad = get_node_attr_s(node, "auto_pad");
            std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
            std::vector<int> dilations = get_node_attr_ai(node, "dilations");
            std::vector<int> strides = get_node_attr_ai(node, "strides");
            std::vector<int> output_padding = get_node_attr_ai(node, "output_padding");
            std::vector<int> output_shape = get_node_attr_ai(node, "output_shape");
            std::vector<int> pads = get_node_attr_ai(node, "pads");
            int num_filter = W.dims(1) * group;

            // filter number
            attributes += "0=" + std::to_string(num_filter);

            // kernel shape
            if (kernel_shape.size() == 1)
            {
                attributes += " 1=" + std::to_string(kernel_shape[0]);
            }
            else if (kernel_shape.size() == 2)
            {
                attributes += " 1=" + std::to_string(kernel_shape[1]);
                attributes += " 11=" + std::to_string(kernel_shape[0]);
            }

            // dilation size
            if (dilations.size() == 1)
            {
                attributes += " 2=" + std::to_string(dilations[0]);
            }
            else if (dilations.size() == 2)
            {
                attributes += " 2=" + std::to_string(dilations[1]);
                attributes += " 12=" + std::to_string(dilations[0]);
            }

            // stride size
            if (strides.size() == 1)
            {
                attributes += " 3=" + std::to_string(strides[0]);
            }
            else if (strides.size() == 2)
            {
                attributes += " 3=" + std::to_string(strides[1]);
                attributes += " 13=" + std::to_string(strides[0]);
            }

            // auto pad
            if (auto_pad == "SAME_UPPER")
            {
                attributes += " 4=-233";
            }
            else if (auto_pad == "SAME_LOWER")
            {
                attributes += " 4=-234";
            }
            else
            {
                if (pads.size() == 1)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                }
                else if (pads.size() == 2)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                    attributes += " 14=" + std::to_string(pads[0]);
                }
                else if (pads.size() == 4)
                {
                    attributes += " 4=" + std::to_string(pads[1]);
                    attributes += " 14=" + std::to_string(pads[0]);
                    attributes += " 15=" + std::to_string(pads[3]);
                    attributes += " 16=" + std::to_string(pads[2]);
                }
            }

            if (output_padding.size() == 1)
            {
                attributes += " 18=" + std::to_string(output_padding[0]);
            }
            else if (output_padding.size() == 2)
            {
                attributes += " 18=" + std::to_string(output_padding[1]);
                attributes += " 19=" + std::to_string(output_padding[0]);
            }

            if (output_shape.size() == 1)
            {
                attributes += " 20=" + std::to_string(output_shape[0]);
            }
            else if (output_shape.size() == 2)
            {
                attributes += " 20=" + std::to_string(output_shape[1]);
                attributes += " 21=" + std::to_string(output_shape[0]);
            }

            attributes += " 5=" + std::to_string(has_bias);
            attributes += " 6=" + std::to_string(get_tensor_proto_data_size(W));

            if (group > 1)
            {
                attributes += " 7=" + std::to_string(group);
            }
            
            int maxk = 0;
            if (kernel_shape.size() == 2)
            {
                maxk = kernel_shape[1] * kernel_shape[0];
            }
            else
            {
                maxk = kernel_shape[0] * kernel_shape[0];
            }
            int weight_data_size = get_tensor_proto_data_size(W);
            const float* weight_data = 0;
            if (W.raw_data().size())
            {
                weight_data = (const float*)W.raw_data().data();
            }
            else if (W.data_type() == 1)
            {
                weight_data = W.float_data().data();
            }
            for (int g = 0; g < group; g++)
            {
                // reorder weight from inch-outch to outch-inch
                int num_filter_g = num_filter / group;
                int num_input = weight_data_size / maxk / num_filter_g / group;
                const float* weight_data_ptr = weight_data + g * maxk * num_filter_g * num_input;
                for (int k = 0; k < num_filter_g; k++)
                {
                    for (int j = 0; j < num_input; j++)
                    {
                        bofs.write((const char*)weight_data_ptr + (j * num_filter_g + k) * maxk, weight_data_size);
                    }
                }
            }
            if (has_bias)
            {
                const onnx::TensorProto& B = weights[node.input(2)];
                ofstream_tensor_proto_data(B, bofs);
            }
        }
        else if (op == "Cos")
        {
            tinyinfer_op_name = "UnaryOp";
            int op_type = 10;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Div")
        {
            tinyinfer_op_name = "BinaryOp";
            int op_type = 3;
            attributes += " 0=" + std::to_string(op_type);
        }
        else if (op == "Dropout")
        {
            tinyinfer_op_name = "Dropout";
            output_size = 1;
        }
        else if (op == "Elu")
        {
            tinyinfer_op_name =  "ELU";
            float alpha = get_node_attr_f(node, "alpha", 1.f);
            attributes += "0=" + std::to_string(alpha);
        }
        else if (op == "Exp")
        {
            tinyinfer_op_name = "UnaryOp";
            int op_type = 7;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Flatten")
        {
            tinyinfer_op_name = "Flatten";
            int axis = get_node_attr_i(node, "axis", 1);
            if (axis != 1)
            {
                fprintf(stderr, "Unsupported Flatten axis %d!\n", axis);
            }
        }
        else if (op == "Gelu")
        {
            tinyinfer_op_name = "Flatten";
            attributes += "0=1";
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
                tinyinfer_op_name = "InnerProduct";

                const onnx::TensorProto& B = weights[node.input(1)];
                const onnx::TensorProto& C = weights[node.input(2)];
                attributes += "0=" + std::to_string(get_tensor_proto_data_size(C));
                attributes += " 1=1";
                attributes += " 2=" + std::to_string(get_tensor_proto_data_size(B));

                ofstream_tensor_proto_data(B, bofs);
                ofstream_tensor_proto_data(C, bofs);
            }
            else
            {
                tinyinfer_op_name = "Gemm";

                attributes += "0=" + std::to_string(alpha);
                attributes += " 1=" + std::to_string(beta);
                attributes += " 2=" + std::to_string(transA);
                attributes += " 3=" + std::to_string(transB);
            }
        }
        else if (op == "GlobalAveragePool")
        {
            tinyinfer_op_name = "Pooling";

            int pool = 1;
            int global_pool = 1;

            attributes += "0=" + std::to_string(pool);
            attributes += " 4=" + std::to_string(global_pool);
        }
        else if (op == "GlobalMaxPool")
        {
            tinyinfer_op_name = "Pooling";

            int pool = 0;
            int global_pool = 1;

            attributes += "0=" + std::to_string(pool);
            attributes += " 4=" + std::to_string(global_pool);
        }
        else if (op == "adaptive_avg_pool2d" || op == "adaptive_max_pool2d")
        {
            tinyinfer_op_name = "Pooling";

            int pool = 0;
            if (op == "adaptive_avg_pool2d")
            {
                pool = 1;
            }
            int adaptive_pooling = 1;
            const onnx::TensorProto& out_shape_tp = weights[node.input(1)];
            std::vector<int> out_shape = get_node_attr_from_input_ai(out_shape_tp);

            attributes += "0=" + std::to_string(pool);
            attributes += " 7=" + std::to_string(pool);
            if (out_shape.size() == 1)
            {
                attributes += " 8=" + std::to_string(out_shape[0]);
            }
            else if (out_shape.size() == 2)
            {
                attributes += " 8=" + std::to_string(out_shape[1]);
                attributes += " 18=" + std::to_string(out_shape[0]);
            }
        }
        else if (op == "HardSigmoid")
        {
            tinyinfer_op_name = "HardSigmoid";

            float alpha = get_node_attr_f(node, "alpha", 0.2f);
            float beta = get_node_attr_f(node, "beta", 0.5f);
            
            attributes += "0=" + std::to_string(alpha);
            attributes += " 1=" + std::to_string(beta);
        }
        else if (op == "HardSwish")
        {
            tinyinfer_op_name = "HardSwish";

            float alpha = get_node_attr_f(node, "alpha", 0.2f);
            float beta = get_node_attr_f(node, "beta", 0.5f);
            attributes += "0=" + std::to_string(alpha);
            attributes += " 1=" + std::to_string(beta);
        }
        else if (op == "Log")
        {
            tinyinfer_op_name = "UnaryOp";
            int op_type = 8;
            attributes += " 0=" + std::to_string(op_type);
        }
        else if (op == "MatMul")
        {
            if (weights.find(node.input(1)) != weights.end() && weights[node.input(1)].dims_size() == 2)
            {
                tinyinfer_op_name = "InnerProduct";

                const onnx::TensorProto& B = weights[node.input(1)];
                int weight_data_size = get_tensor_proto_data_size(B);
                int num_output = B.dims(B.dims_size() - 1);
                int num_input = weight_data_size / num_output;

                attributes += "0=" + std::to_string(num_output);
                attributes += " 1=0";
                attributes += " 2=" + std::to_string(weight_data_size);

                {
                    const float* bptr = B.raw_data().size() ? (const float*)B.raw_data().data() : B.float_data().data();

                    for (int j = 0; j < num_output; j++)
                    {
                        for (int k = 0; k < num_input; k++)
                        {
                            float vb = bptr[k * num_output + j];
                            bofs.write((const char*)&vb, sizeof(float));
                        }
                    }
                }
            }
            else
            {
                tinyinfer_op_name = "Gemm";
            }
        }
        else if (op == "Max")
        {
            tinyinfer_op_name = "BinaryOp";

            int op_type = 4;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Min")
        {
            tinyinfer_op_name = "BinaryOp";
            
            int op_type = 5;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Mul")
        {
            tinyinfer_op_name = "BinaryOp";

            int op_type = 2;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Pad")
        {
            tinyinfer_op_name = "BinaryOp";

            std::string mode = get_node_attr_s(node, "mode");
            float value = get_node_attr_f(node, "value", 0.f);

            std::vector<int> pads;
            if (node.input_size() == 1)
            {
                pads = get_node_attr_ai(node, "pads");
            }
            else
            {
                pads = get_node_attr_from_input_ai(weights[node.input(1)]);
            }

            int type = 0;
            if (mode == "constant")
            {
                type = 0;
            }
            else if (mode == "edge")
            {
                type = 1;
            }
            else if (mode == "reflect")
            {
                type = 2;
            }

            int pad_size = (int)pads.size();
            int top = 0;
            int bottom = 0;
            int left = 0;
            int right = 0;
            int front = 0;
            int behind = 0;
            if (pad_size == 8)
            {
                //NCHW
                top = pads[2];
                bottom = pads[6];
                left = pads[3];
                right = pads[7];
                front = pads[1];
                behind = pads[5];
            }
            else if (pad_size == 6)
            {
                //NHW
                top = pads[1];
                bottom = pads[4];
                left = pads[2];
                right = pads[5];
            }
            else
            {
                //NW
                left = pads[1];
                right = pads[3];
            }

            attributes += "0=" + std::to_string(top);
            attributes += " 1=" + std::to_string(bottom);
            attributes += " 2=" + std::to_string(left);
            attributes += " 3=" + std::to_string(right);
            attributes += " 4=" + std::to_string(type);
            attributes += " 5=" + std::to_string(value);
            attributes += " 6=" + std::to_string(front);
            attributes += " 7=" + std::to_string(behind);
        }
        else if (op == "Relu")
        {
            tinyinfer_op_name = "RelU";
        }
        else if (op == "Reshape")
        {
            tinyinfer_op_name = "Reshape";

            std::vector<int> shape;

            if (node.input_size() == 1)
            {
                shape = get_node_attr_ai(node, "shape");
            }
            else
            {
                shape = get_node_attr_from_input_ai(weights[node.input(1)]);
            }

            if (shape.size() == 1)
            {
                attributes += "0=" + std::to_string(shape[0]); // should never reach here
            }
            else if (shape.size() == 2)
            {
                attributes += "0=" + std::to_string(shape[1]);
            }
            else if (shape.size() == 3)
            {
                attributes += "0=" + std::to_string(shape[2]);
                attributes += " 1=" + std::to_string(shape[1]);
            }
            else if (shape.size() == 4)
            {
                attributes += "0=" + std::to_string(shape[3]);
                attributes += " 1=" + std::to_string(shape[2]);
                attributes += " 2=" + std::to_string(shape[1]);
            }
            else if (shape.size() == 5)
            {
                attributes += "0=" + std::to_string(shape[4] * shape[3]);
                attributes += " 1=" + std::to_string(shape[2]);
                attributes += " 2=" + std::to_string(shape[1]);
            }
        }
        else if (op == "Sigmoid")
        {
            tinyinfer_op_name = "Sigmoid";
        }
        else if (op == "Softmax")
        {
            tinyinfer_op_name = "Softmax";
            int axis = get_node_attr_i(node, "axis", 1);
            attributes += "0=" + std::to_string(axis - 1);
            attributes += " 1=1";
        }
        else if (op == "Squeeze")
        {
            tinyinfer_op_name = "Squeeze";
            
            std::vector<int> axes = get_node_attr_ai(node, "axes");
            if (axes.empty())
            {
                attributes += "0=1 1=1 2=1";
            }
            else
            {
                attributes += "-23303=" + std::to_string(axes.size());
                for (int i = 0; i < (int)axes.size(); i++)
                {
                    if (axes[i] == 0 || axes[i] > 4 || axes[i] < -3)
                        fprintf(stderr, "Unsupported squeeze axes !\n");
                    attributes += "," + std::to_string(axes[i] > 0 ? axes[i] - 1 : axes[i]);
                }
            }
        }
        else if (op == "Sum")
        {
            tinyinfer_op_name = "Sum";

            int op_type = 1;
            attributes += "0=" + std::to_string(op_type);
        }
        else if (op == "Swish")
        {
            tinyinfer_op_name = "Swish";
        }
        else if (op == "Transpose")
        {
            tinyinfer_op_name = "Permute";
        }
        else if (op == "Upsample" || op == "Resize")
        {
            tinyinfer_op_name = "Interp";
        }
        else if (op == "Unsqueeze")
        {
            tinyinfer_op_name = "ExpandDims";
        }
        else
        {
            fprintf(stderr, "%s not support yet!\n", op.c_str());
            tinyinfer_op_name = op;
        }
        
        // [input_names]
        std::string input_names;
        for (int j = 0; j < (int)node.input_size(); j++)
        {
            std::string input_name = node.input(j);

            if (weights.find(input_name) != weights.end() && node_reference_cnt[input_name] == 0)
                continue;
            if (input_name.empty())
                continue;

            if (split_node_reference.find(input_name) != split_node_reference.end())
            {
                int refidx = split_node_reference[input_name] - 1;
                split_node_reference[input_name] = refidx;

                std::string split_suffix = "_split_" + std::to_string(refidx);
                input_name = input_name + split_suffix;
            }
            input_names += input_name;
            if (j < input_size - 1) input_names += " ";
        }


        // [output_names]
        std::string output_names;
        for (int j = 0; j < output_size; j++)
        {
            const std::string& output_name = node.output(j);
            output_names += output_name;
            if (j < output_size - 1) output_names += " ";
        }

        pofs << std::left << std::setw(16) << tinyinfer_op_name;
        pofs << " " << std::left << std::setw(24) << node_name;
        pofs << " " << input_size << " " << output_size;
        pofs << " " << input_names << " " << output_names;
        pofs << " " << attributes << std::endl;

        for (int j = 0; j < output_size; j++)
        {
            const std::string& output_name = node.output(j);
            if (node_reference_cnt.find(output_name) != node_reference_cnt.end())
            {
                int refcount = node_reference_cnt[output_name];
                if (refcount > 1)
                {
                    std::string splitname = "split_tinyinfer_" + std::to_string(internal_split);
                    pofs << std::left << std::setw(16) << "Split";
                    pofs << " " << std::left << std::setw(24) << splitname;
                    pofs << " 1" << " " << refcount;
                    pofs << " " << output_name;

                    for (int k = 0; k < refcount; k++)
                    {
                        std::string split_name = output_name + "_split_" + std::to_string(k);
                        pofs << " " << split_name;
                    }
                    pofs << std::endl;

                    internal_split++;
                }
            }
        }
    }

    pofs.close();
    bofs.close();
}