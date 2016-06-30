#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/picpac_data_layer.hpp"

namespace caffe {

template <typename Dtype>
PicPacLayer<Dtype>::PicPacLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param), fold_(0), K_(1)
{
    config_.shuffle = true;
    config_.stratify = true;
    config_.loop = true;
#define PICPAC_CONFIG_UPDATE(c,v) if (param.picpac_param().has_##v()) { \
        config_.v = param.picpac_param().v(); }
    PICPAC_CONFIG_UPDATE_ALL(0);
#undef PICPAC_CONFIG_UPDATE
    path_ = param.picpac_param().path();
    //stream_ = new picpac::BatchImageStream(param.picpac_param().path(), config);
}

template <typename Dtype>
PicPacLayer<Dtype>::~PicPacLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void PicPacLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //config_.loop = true;
  if (this->phase_ == TRAIN) {
      //config_.reshuffle = true;
      //config_.shuffle = true;
      //config_.stratify = true;
      config_.split_negate = false;
  }
  else {
      //config_.reshuffle = false;
      //config_.shuffle = true;
      //config_.stratify = true;
      config_.split_negate = true;
      config_.perturb = false;
      config_.mixin.clear();
  }
  stream_ = shared_ptr<picpac::BatchImageStream>(new picpac::BatchImageStream(path_, config_));

  vector<int> images_shape;
  vector<int> labels_shape;
  stream_->next_shape(&images_shape, &labels_shape);
  top[0]->Reshape(images_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(images_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  top[1]->Reshape(labels_shape);
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(labels_shape);
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void PicPacLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  CHECK(batch->data_.count());
  vector<int> images_shape;
  vector<int> labels_shape;
  stream_->next_shape(&images_shape, &labels_shape);

  batch->data_.Reshape(images_shape);
  batch->label_.Reshape(labels_shape);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  stream_->next_fill(top_data, top_label);
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(PicPacLayer);
REGISTER_LAYER_CLASS(PicPac);
#ifdef CPU_ONLY
STUB_GPU_FORWARD(PicPacDataLayer, Forward);
#endif

}  // namespace caffe
