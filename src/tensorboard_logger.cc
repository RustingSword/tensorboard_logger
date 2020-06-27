#include "tensorboard_logger.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "event.pb.h"
#include "projector_config.pb.h"
#include "plugin_pr_curve.pb.h"

using namespace std;
using google::protobuf::TextFormat;
using tensorflow::EmbeddingInfo;
using tensorflow::Event;
using tensorflow::HistogramProto;
using tensorflow::ProjectorConfig;
// using tensorflow::SpriteMetadata;
using tensorflow::Summary;
using tensorflow::SummaryMetadata;
using tensorflow::TensorProto;
using tensorflow::PrCurvePluginData;
using tensorflow::TensorShapeProto;

// https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L115
int TensorBoardLogger::generate_default_buckets(std::vector<double> range,
    size_t num_of_bins,
    bool ignore_outside_range,
    bool regenerate ) {
    if (bucket_limits_ == nullptr || regenerate == true) {
        bucket_limits_ = new vector<double>;
        double v = range[0];
        double width = (range[1] - range[0]) / num_of_bins ;
        if (width == 0)
            width = 1;
        if(!ignore_outside_range)
            bucket_limits_->push_back(numeric_limits<double>::lowest());
        while (v <= range[1]) {
            bucket_limits_->push_back(v);
            v = v + width;
        }
        if(!ignore_outside_range)
        {
            bucket_limits_->push_back(numeric_limits<double>::max());
        }
    }
    return 0;
}

int TensorBoardLogger::add_scalar(const string &tag, int step, double value) {
    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_simple_value(value);
    return add_event(step, summary);
}

int TensorBoardLogger::add_scalar(const string &tag, int step, float value) {
    return add_scalar(tag, step, static_cast<double>(value));
}

int TensorBoardLogger::add_image(const string &tag, int step,
                                 const string &encoded_image, int height,
                                 int width, int channel,
                                 const string &display_name,
                                 const string &description) {
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name == "" ? tag : display_name);
    meta->set_summary_description(description);

    auto *image = new Summary::Image();
    image->set_height(height);
    image->set_width(width);
    image->set_colorspace(channel);
    image->set_encoded_image_string(encoded_image);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_image(image);
    v->set_allocated_metadata(meta);
    return add_event(step, summary);
}

int TensorBoardLogger::add_images(
    const std::string &tag, int step,
    const std::vector<std::string> &encoded_images, int height, int width,
    const std::string &display_name, const std::string &description) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name("images");
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name == "" ? tag : display_name);
    meta->set_summary_description(description);
    meta->set_allocated_plugin_data(plugin_data);

    auto *tensor = new TensorProto();
    tensor->set_dtype(tensorflow::DataType::DT_STRING);
    tensor->add_string_val(to_string(width));
    tensor->add_string_val(to_string(height));
    for (const auto &image : encoded_images) tensor->add_string_val(image);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_tensor(tensor);
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_audio(const string &tag, int step,
                                 const string &encoded_audio, float sample_rate,
                                 int num_channels, int length_frame,
                                 const string &content_type,
                                 const string &display_name,
                                 const string &description) {
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name == "" ? tag : display_name);
    meta->set_summary_description(description);

    auto *audio = new Summary::Audio();
    audio->set_sample_rate(sample_rate);
    audio->set_num_channels(num_channels);
    audio->set_length_frames(length_frame);
    audio->set_encoded_audio_string(encoded_audio);
    audio->set_content_type(content_type);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_audio(audio);
    v->set_allocated_metadata(meta);
    return add_event(step, summary);
}

int TensorBoardLogger::add_text(const string &tag, int step, const char *text) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name(kTextPluginName);

    auto *meta = new SummaryMetadata();
    meta->set_allocated_plugin_data(plugin_data);

    auto *tensor = new TensorProto();
    tensor->set_dtype(tensorflow::DataType::DT_STRING);

    auto *str_val = tensor->add_string_val();
    *str_val = text;

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_tensor(tensor);
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_embedding(const std::string &tensor_name,
                                     const std::string &tensordata_path,
                                     const std::string &metadata_path,
                                     const std::vector<uint32_t> &tensor_shape,
                                     int step) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name(kProjectorPluginName);
    auto *meta = new SummaryMetadata();
    meta->set_allocated_plugin_data(plugin_data);

    const auto &filename = log_dir_ + kProjectorConfigFile;
    auto *conf = new ProjectorConfig();

    // parse possibly existing config file
    ifstream fin(filename);
    if (fin.is_open()) {
        ostringstream ss;
        ss << fin.rdbuf();
        TextFormat::ParseFromString(ss.str(), conf);
        fin.close();
    }

    auto *embedding = conf->add_embeddings();
    embedding->set_tensor_name(tensor_name);
    embedding->set_tensor_path(tensordata_path);
    if (metadata_path != "") {
        embedding->set_metadata_path(metadata_path);
    }
    if (tensor_shape.size() > 0) {
        for (auto shape : tensor_shape) embedding->add_tensor_shape(shape);
    }

    // `conf` and `embedding` will be deleted by ProjectorConfig destructor

    ofstream fout(filename);
    string content;
    TextFormat::PrintToString(*conf, &content);
    fout << content;
    fout.close();

    // Following line is just to add plugin and does not hold any meaning
    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag("embedding");
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_embedding(
    const std::string &tensor_name,
    const std::vector<std::vector<float>> &tensor,
    const std::string &tensordata_filename,
    const std::vector<std::string> &metadata,
    const std::string &metadata_filename, int step) {
    ofstream binary_tensor_file(log_dir_ + tensordata_filename, ios::binary);
    if (!binary_tensor_file.is_open()) {
        throw std::runtime_error("failed to open binary tensor file " +
                                 log_dir_ + tensordata_filename);
    }

    for (const auto &vec : tensor) {
        binary_tensor_file.write(reinterpret_cast<const char *>(vec.data()),
                                 vec.size() * sizeof(float));
    }
    binary_tensor_file.close();
    if (metadata.size() > 0) {
        if (metadata.size() != tensor.size()) {
            throw std::runtime_error("tensor size != metadata size");
        }
        ofstream metadata_file(log_dir_ + metadata_filename);
        if (!metadata_file.is_open()) {
            throw std::runtime_error("failed to open metadata file " +
                                     log_dir_ + metadata_filename);
        }
        for (const auto &meta : metadata) metadata_file << meta << endl;
        metadata_file.close();
    }
    vector<uint32_t> tensor_shape;
    tensor_shape.push_back(tensor.size());
    tensor_shape.push_back(tensor[0].size());
    return add_embedding(tensor_name, tensordata_filename, metadata_filename,
                         tensor_shape, step);
}

std::vector<std::vector<double>> TensorBoardLogger::compute_curve(
    const std::vector<double>labels,
    const std::vector<double>predictions,
    int num_thresholds,
    std::vector<double>weights)
{
    // misbheaves when thresholds is greater than 127
    num_thresholds = min(num_thresholds,127);
    double min_count = 1e-7;
    std::vector<std::vector<double>> data;
    while (weights.size()<labels.size())
    {
        weights.push_back(1.0);
    }
    generate_default_buckets({0, (double)num_thresholds - 1}, num_thresholds, true, true);
    vector<double> tp(bucket_limits_->size(), 0), fp(bucket_limits_->size(), 0);
    
    for (size_t i = 0; i < labels.size(); ++i)
    {
        float v = labels[i];
        int item = predictions[i] * (num_thresholds -1);
        auto lb =
            lower_bound(bucket_limits_->begin(), bucket_limits_->end(), item);
        {
            tp[lb - bucket_limits_->begin()] = tp[lb - bucket_limits_->begin()] + (v*weights[i]);
            fp[lb - bucket_limits_->begin()] = fp[lb - bucket_limits_->begin()] + ((1-v)*weights[i]);
        }
    }

    // Reverse cummulative sum 
    for(int i = tp.size() - 1; i >= 0 ;i--)
    {
        tp[i] = tp[i] + tp[i+1];
        fp[i] = fp[i] + fp[i+1];
    }
    reverse(tp.begin(), tp.end());
    reverse(fp.begin(), fp.end());
    for(int i = tp.size() - 1; i >= 0 ;i--)
    {

        tp[i] = tp[i] + tp[i+1];
        fp[i] = fp[i] + fp[i+1];
    }
    std::vector<double> tn(tp.size()), fn(tp.size()), precision(tp.size()), recall(tp.size());
    for(size_t i = 0; i < tp.size() ;i++)
    {
        tn[i] = tp[0] - tp[i];
        fn[i] = fp[0] - fp[i];
        precision[i] = tp[i] / max(min_count,tp[i]+fp[i]);
        recall[i] = tp[i] / max(min_count,tp[i]+fn[i]);
    }
    data.push_back(tp);
    data.push_back(fp);
    data.push_back(tn);
    data.push_back(fn);
    data.push_back(precision);
    data.push_back(recall);
    return data;
}
int TensorBoardLogger::prcurve(
    const std::string tag,
    const std::vector<double>labels, 
    const std::vector<double>predictions, 
    const int num_thresholds,
    std::vector<double>weights,
    const std::string &display_name,
    const std::string &description)
{
    // Pr plugin
    PrCurvePluginData *pr_curve_plugin = new PrCurvePluginData();
    pr_curve_plugin->set_version(0);
    pr_curve_plugin->set_num_thresholds(num_thresholds);
    std::string pr_curve_content;
    pr_curve_plugin->SerializeToString(&pr_curve_content);

    // PluginMeta data
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name("pr_curves");
    plugin_data->set_content(pr_curve_content);

    // Summary Meta data
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name == "" ? tag : display_name);
    meta->set_summary_description(description);
    meta->set_allocated_plugin_data(plugin_data);

    std::vector<std::vector<double>> data =
        compute_curve(labels, predictions, num_thresholds, weights);

    // Prepare Tensor
    auto *tensorshape = new TensorShapeProto();
    auto rowdim = tensorshape->add_dim();
    rowdim->set_size(data.size());
    auto coldim = tensorshape->add_dim();
    coldim->set_size(data[0].size());   
    auto *tensor = new TensorProto();
    tensor->set_dtype(tensorflow::DataType::DT_DOUBLE);
    tensor->set_allocated_tensor_shape(tensorshape);
    for(int i=0;i<data.size();i++)
    {
        for(int j=0;j<data[0].size();j++)
        {
            tensor->add_double_val(data[i][j]);
        }
    }

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_tensor(tensor);
    v->set_allocated_metadata(meta);

    return add_event(0, summary);    
}

int TensorBoardLogger::add_embedding(const std::string &tensor_name,
                                     const float *tensor,
                                     const std::vector<uint32_t> &tensor_shape,
                                     const std::string &tensordata_filename,
                                     const std::vector<std::string> &metadata,
                                     const std::string &metadata_filename,
                                     int step) {
    ofstream binary_tensor_file(log_dir_ + tensordata_filename, ios::binary);
    if (!binary_tensor_file.is_open()) {
        throw std::runtime_error("failed to open binary tensor file " +
                                 log_dir_ + tensordata_filename);
    }

    uint32_t num_elements = 1;
    for (auto shape : tensor_shape) num_elements *= shape;
    binary_tensor_file.write(reinterpret_cast<const char *>(tensor),
                             num_elements * sizeof(float));
    binary_tensor_file.close();
    if (metadata.size() > 0) {
        if (metadata.size() != tensor_shape[0]) {
            throw std::runtime_error("tensor size != metadata size");
        }
        ofstream metadata_file(log_dir_ + metadata_filename);
        if (!metadata_file.is_open()) {
            throw std::runtime_error("failed to open metadata file " +
                                     log_dir_ + metadata_filename);
        }
        for (const auto &meta : metadata) metadata_file << meta << endl;
        metadata_file.close();
    }
    return add_embedding(tensor_name, tensordata_filename, metadata_filename,
                         tensor_shape, step);
}

int TensorBoardLogger::add_event(int64_t step, Summary *summary) {
    Event event;
    double wall_time = time(nullptr);
    event.set_wall_time(wall_time);
    event.set_step(step);
    event.set_allocated_summary(summary);
    return write(event);
}

int TensorBoardLogger::write(Event &event) {
    string buf;
    event.SerializeToString(&buf);
    auto buf_len = static_cast<uint64_t>(buf.size());
    uint32_t len_crc =
        masked_crc32c((char *)&buf_len, sizeof(buf_len));  // NOLINT
    uint32_t data_crc = masked_crc32c(buf.c_str(), buf.size());

    ofs_->write((char *)&buf_len, sizeof(buf_len));  // NOLINT
    ofs_->write((char *)&len_crc, sizeof(len_crc));  // NOLINT
    ofs_->write(buf.c_str(), buf.size());
    ofs_->write((char *)&data_crc, sizeof(data_crc));  // NOLINT
    ofs_->flush();
    return 0;
}

string get_parent_dir(const string &path) {
    auto last_slash_pos = path.find_last_of("/\\");
    if (last_slash_pos == string::npos) {
        return "./";
    }
    return path.substr(0, last_slash_pos + 1);
}
