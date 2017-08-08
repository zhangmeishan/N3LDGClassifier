
include_directories(
  model
)

add_executable(NNLSTMLabeler NNLSTMLabeler.cpp)
target_link_libraries(NNLSTMLabeler ${LIBS})
