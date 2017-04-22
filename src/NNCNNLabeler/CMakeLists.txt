
include_directories(
  model
)

add_executable(NNCNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNLabeler ${LIBS})
