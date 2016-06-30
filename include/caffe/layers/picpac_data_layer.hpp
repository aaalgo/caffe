#ifndef CAFFE_HDF5_DATA_LAYER_HPP_
#define CAFFE_HDF5_DATA_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>
#include <picpac.h>
#include <picpac-cv.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from PicPac image database.
 *
 */
template <typename Dtype>
class PicPacLayer : public BasePrefetchingDataLayer<Dtype> {
    string path_;
    unsigned fold_, K_; 
    picpac::BatchImageStream::Config config_;
 public:
  explicit PicPacLayer(const LayerParameter& param);
  virtual ~PicPacLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  shared_ptr<picpac::BatchImageStream> stream_;
};


}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
