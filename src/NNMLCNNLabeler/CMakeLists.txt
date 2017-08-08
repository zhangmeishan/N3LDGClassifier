
include_directories(
  model
)

add_executable(NNMLCNNLabeler NNMLCNNLabeler.cpp)
target_link_libraries(NNMLCNNLabeler ${LIBS})
