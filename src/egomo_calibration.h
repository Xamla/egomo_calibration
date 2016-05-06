#ifndef egomo_calib_h
#define egomo_calib_h

extern "C" {
#include <TH/TH.h>
}

#include <stdexcept>

#define CALIBIMP(return_type, class_name, name) extern "C" return_type TH_CONCAT_4(calib_, class_name, _, name)

/*class EgomoCalibWrapperException
  : public std::runtime_error {
public:
  RosWrapperException(const std::string& reason)
    : runtime_error(reason) {
  }
};*/

#endif  // egomo_calib_h
