#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <atomic>
#include <exception>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "crc.h"
#include "event.pb.h"
#include "plugin_data.pb.h"
using ::google::protobuf::Value;
using std::map;
using std::string;
using tensorboard::hparams::HParamsPluginData;
using tensorboard::hparams::SessionStartInfo;
using tensorflow::Event;
using tensorflow::Summary;

// extract parent dir or basename from path by finding the last slash
std::string get_parent_dir(const std::string &path);
std::string get_basename(const std::string &path);

const std::string kProjectorConfigFile = "projector_config.pbtxt";
const std::string kProjectorPluginName = "projector";
const std::string kTextPluginName = "text";

struct TensorBoardLoggerOptions {
    // Log is flushed whenever this many entries have been written since the
    // last forced flush.
    size_t max_queue_size_ = 100000;
    TensorBoardLoggerOptions &max_queue_size(size_t max_queue_size) {
        max_queue_size_ = max_queue_size;
        return *this;
    }

    // Log is flushed with this period.
    size_t flush_period_s_ = 60;
    TensorBoardLoggerOptions &flush_period_s(size_t flush_period_s) {
        flush_period_s_ = flush_period_s;
        return *this;
    }

    bool resume_ = false;
    TensorBoardLoggerOptions &resume(bool resume) {
        resume_ = resume;
        return *this;
    }
};

class TensorBoardLogger {
   public:
    explicit TensorBoardLogger(const std::string &log_file,
                               const TensorBoardLoggerOptions &options = {}) {
        this->options = options;
        auto basename = get_basename(log_file);
        if (basename.find("tfevents") == std::string::npos) {
            throw std::runtime_error(
                "A valid event file must contain substring \"tfevents\" in its "
                "basename, got " +
                basename);
        }
        bucket_limits_ = nullptr;
        ofs_ = new std::ofstream(
            log_file, std::ios::out |
                          (options.resume_ ? std::ios::app : std::ios::trunc) |
                          std::ios::binary);
        if (!ofs_->is_open()) {
            throw std::runtime_error("failed to open log_file " + log_file);
        }
        log_dir_ = get_parent_dir(log_file);

        flushing_thread = std::thread(&TensorBoardLogger::flusher, this);
    }
    ~TensorBoardLogger() {
        ofs_->close();
        delete ofs_;
        if (bucket_limits_ != nullptr) {
            delete bucket_limits_;
            bucket_limits_ = nullptr;
        }

        stop = true;
        if (flushing_thread.joinable()) {
            flushing_thread.join();
        }
    }

    int add_hparams(const map<string, Value> &hparams, const string &group_name,
                    double start_time_secs);
    int add_scalar(const std::string &tag, int step, double value);
    int add_scalar(const std::string &tag, int step, float value);

    // https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L127
    template <typename T>
    int add_histogram(const std::string &tag, int step, const T *value,
                      size_t num) {
        if (bucket_limits_ == nullptr) {
            generate_default_buckets();
        }

        std::vector<int> counts(bucket_limits_->size(), 0);
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::lowest();
        double sum = 0.0;
        double sum_squares = 0.0;
        for (size_t i = 0; i < num; ++i) {
            T v = value[i];
            auto lb = std::lower_bound(bucket_limits_->begin(),
                                       bucket_limits_->end(), v);
            counts[lb - bucket_limits_->begin()]++;
            sum += v;
            sum_squares += v * v;
            if (v > max) {
                max = v;
            } else if (v < min) {
                min = v;
            }
        }

        auto *histo = new tensorflow::HistogramProto();
        histo->set_min(min);
        histo->set_max(max);
        histo->set_num(num);
        histo->set_sum(sum);
        histo->set_sum_squares(sum_squares);
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                histo->add_bucket_limit((*bucket_limits_)[i]);
                histo->add_bucket(counts[i]);
            }
        }

        auto *summary = new tensorflow::Summary();
        auto *v = summary->add_value();
        v->set_tag(tag);
        v->set_allocated_histo(histo);

        return add_event(step, summary);
    };

    template <typename T>
    int add_histogram(const std::string &tag, int step,
                      const std::vector<T> &values) {
        return add_histogram(tag, step, values.data(), values.size());
    };

    // metadata (such as display_name, description) of the same tag will be
    // stripped to keep only the first one.
    int add_image(const std::string &tag, int step,
                  const std::string &encoded_image, int height, int width,
                  int channel, const std::string &display_name = "",
                  const std::string &description = "");
    int add_images(const std::string &tag, int step,
                   const std::vector<std::string> &encoded_images, int height,
                   int width, const std::string &display_name = "",
                   const std::string &description = "");
    int add_audio(const std::string &tag, int step,
                  const std::string &encoded_audio, float sample_rate,
                  int num_channels, int length_frame,
                  const std::string &content_type,
                  const std::string &display_name = "",
                  const std::string &description = "");
    int add_text(const std::string &tag, int step, const char *text);

    // `tensordata` and `metadata` should be in tsv format, and should be
    // manually created before calling `add_embedding`
    //
    // `tensor_name` is mandated to differentiate tensors
    //
    // TODO add sprite image support
    int add_embedding(
        const std::string &tensor_name, const std::string &tensordata_path,
        const std::string &metadata_path = "",
        const std::vector<uint32_t> &tensor_shape = std::vector<uint32_t>(),
        int step = 1 /* no effect */);
    // write tensor to binary file
    int add_embedding(
        const std::string &tensor_name,
        const std::vector<std::vector<float>> &tensor,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);
    int add_embedding(
        const std::string &tensor_name, const float *tensor,
        const std::vector<uint32_t> &tensor_shape,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);

   private:
    int generate_default_buckets();
    int add_session_start_info(SessionStartInfo *session_start_info);
    int add_event(int64_t step, Summary *summary);
    int write(Event &event);
    void flusher();

    std::string log_dir_;
    std::ofstream *ofs_;
    std::vector<double> *bucket_limits_;
    TensorBoardLoggerOptions options;

    std::atomic<bool> stop{false};
    size_t queue_size{0};
    std::thread flushing_thread;
    std::mutex file_object_mtx{};
};  // class TensorBoardLogger

#endif  // TENSORBOARD_LOGGER_H
