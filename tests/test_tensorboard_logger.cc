#include <random>
#include <vector>
#include "tensorboard_logger.h"

using namespace std;

int test_log(const char *log_file) {
    TensorBoardLogger logger(log_file);
    default_random_engine generator;
    normal_distribution<double> default_distribution(0, 1.0);
    for (int i = 0; i < 10; ++i) {
        normal_distribution<double> distribution(i * 0.1, 1.0);
        vector<float> values;
        for (int j = 0; j < 10000; ++j)
            values.push_back(distribution(generator));
        logger.add_histogram("param", i, values);
        logger.add_scalar("acc", i, default_distribution(generator));
    }
    return 0;
}

int main(int argc, char* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    int ret = test_log("demo/tfevents.pb");
    assert(ret == 0);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

