add_executable(example_hello_world
  example_hello_world.cpp
)

target_link_libraries(example_hello_world PRIVATE yorch::yorch)
