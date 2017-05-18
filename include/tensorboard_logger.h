#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <fstream>
#include <string>
#include <vector>
#include "event.pb.h"
#include "crc.h"

using tensorboard::Summary;
using tensorboard::Event;

class TensorBoardLogger {
public:
    explicit TensorBoardLogger(const char *log_file) {
        bucket_limits_ = NULL;
        ofs_ = new std::ofstream(log_file,
                std::ios::out | std::ios::trunc | std::ios::binary);
    }
    ~TensorBoardLogger() {
        ofs_->close();
        if (bucket_limits_ != NULL) {
            delete bucket_limits_;
            bucket_limits_ = NULL;
        }
    }
    int add_scalar(const std::string &tag, int step, float value);
    int add_histogram(const std::string &tag, int step, std::vector<float> &value);

private:
    int generate_default_buckets();
    int add_event(int64_t step, Summary *summary);
    int write(Event &event);

    std::ofstream *ofs_;
    std::vector<double> *bucket_limits_;
};  // class TensorBoardLogger
#endif  // TENSORBOARD_LOGGER_H
