#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "tensorboard_logger.h"

using namespace std;

int test_log(const char* log_file) {
    TensorBoardLogger logger(log_file);

    // test scalar and histogram
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

    // test add image
    ifstream fin("./assets/Lenna_(test_image).png", ios::binary);
    ostringstream ss;

    ss << fin.rdbuf();
    string image1(ss.str());
    ss.str("");
    fin.close();
    fin.open("./assets/audio.jpg", ios::binary);
    ss << fin.rdbuf();
    string image2(ss.str());
    ss.str("");
    fin.close();

    // add single image
    logger.add_image("single image", 1, image1, 512, 512, 3, "Lena", "Lena");
    logger.add_image("single image", 2, image2, 512, 512, 3, "TensorBoard",
                     "Text");

    // add multiple images
    logger.add_images(
        "Image Sample", 1, {image1, image2}, 512, 512, "Lena Forsén",
        "Lenna or Lena is the name given to a standard test image widely used "
        "in the field of image processing since 1973.");
    logger.add_images(
        "Image Sample", 2, {image2, image1}, 512, 512, "Lena Forsén",
        "Lenna or Lena is the name given to a standard test image widely used "
        "in the field of image processing since 1973.");

    // test add audio
    fin.open("./assets/file_example_WAV_1MG.wav", ios::binary);
    ss << fin.rdbuf();
    string audio(ss.str());
    fin.close();
    ss.str("");
    logger.add_audio("Audio Sample", 1, audio, 8000, 2, 8000 * 16 * 2 * 33,
                     "audio/wav", "Impact Moderato",
                     "https://file-examples.com/index.php/sample-audio-files/"
                     "sample-wav-download/");

    // test add text
    logger.add_text("Text Sample", 1, "Hello World");

    // test add embedding
    logger.add_embedding("vocab", "../assets/vecs.tsv", "../assets/meta.tsv");
    logger.add_embedding("another vocab without labels", "../assets/vecs.tsv");

    // test add binary embedding
    vector<vector<float>> tensor;
    string line;
    ifstream vec_file("assets/vecs.tsv");
    uint32_t num_elements = 1;
    while (getline(vec_file, line)) {
        istringstream values(line);
        vector<float> vec;
        copy(istream_iterator<float>(values), istream_iterator<float>(),
             back_inserter(vec));
        num_elements += vec.size();
        tensor.push_back(vec);
    }
    vec_file.close();

    vector<string> meta;
    ifstream meta_file("assets/meta.tsv");
    while (getline(meta_file, line)) {
        meta.push_back(line);
    }
    meta_file.close();
    logger.add_embedding("binary tensor", tensor, "tensor.bin", meta,
                         "binary_tensor.tsv");

    // test tensor stored as 1d array
    float* tensor_1d = new float[num_elements];
    for (size_t i = 0; i < tensor.size(); i++) {
        const auto& vec = tensor[i];
        memcpy(tensor_1d + i * vec.size(), vec.data(),
               vec.size() * sizeof(float));
    }
    vector<uint32_t> tensor_shape;
    tensor_shape.push_back(tensor.size());
    tensor_shape.push_back(tensor[0].size());
    logger.add_embedding("binary tensor 1d", tensor_1d, tensor_shape,
                         "tensor_1d.bin", meta, "binary_tensor_1d.tsv");
    delete[] tensor_1d;

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
