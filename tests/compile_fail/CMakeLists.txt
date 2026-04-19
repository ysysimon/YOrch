function(yorch_expect_compile_failure case_name expected_message failure_message)
  try_compile(
    yorch_compile_failed
    "${CMAKE_CURRENT_BINARY_DIR}/${case_name}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${case_name}"
    "${case_name}"
    OUTPUT_VARIABLE yorch_compile_output
  )

  if(yorch_compile_failed)
    message(FATAL_ERROR
      "Expected ${failure_message} compile-fail test to be rejected, but it compiled successfully.")
  endif()

  string(FIND
    "${yorch_compile_output}"
    "${expected_message}"
    yorch_message_pos)

  if(yorch_message_pos EQUAL -1)
    message(FATAL_ERROR
      "Expected ${failure_message} compile-fail test to mention the expected diagnostic, but compiler output was:\n${yorch_compile_output}")
  endif()
endfunction()

yorch_expect_compile_failure(
  task_result_void
  "yorch::task_result<void> is not supported"
  "yorch::task_result<void>"
)

yorch_expect_compile_failure(
  forward_prev_parent_type_mismatch
  "borrow_prev_mut<T>() requires a direct parent slot carrying the requested type"
  "forward_prev parent-type mismatch"
)
