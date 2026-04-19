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
  bind_member_callable
  "bind(...) does not accept member function pointers; use bind_member(...) instead"
  "bind member callable"
)

yorch_expect_compile_failure(
  bind_callable_shape_invalid
  "bind(...) requires a callable with one non-overloaded concrete signature"
  "bind callable shape invalid"
)

yorch_expect_compile_failure(
  bind_arity_mismatch
  "bind(...) requires exactly one spec per function parameter"
  "bind arity mismatch"
)

yorch_expect_compile_failure(
  bind_into_void_output
  "bind_into<T>(...) requires a non-void owned payload type T"
  "bind_into void output"
)

yorch_expect_compile_failure(
  bind_into_reference_output
  "bind_into<T>(...) requires a non-void owned payload type T"
  "bind_into reference output"
)

yorch_expect_compile_failure(
  bind_into_member_callable
  "bind_into(...) does not accept member function pointers; use bind_into_member(...) instead"
  "bind_into member callable"
)

yorch_expect_compile_failure(
  bind_into_callable_shape_invalid
  "bind_into(...) requires a callable with one non-overloaded concrete signature"
  "bind_into callable shape invalid"
)

yorch_expect_compile_failure(
  bind_into_missing_output_parameter
  "bind_into(...) requires a callable whose last parameter is yorch::direct_out<T>"
  "bind_into missing output parameter"
)

yorch_expect_compile_failure(
  bind_into_arity_mismatch
  "bind_into(...) requires exactly one spec per non-output function parameter"
  "bind_into arity mismatch"
)

yorch_expect_compile_failure(
  bind_into_wrong_output_parameter
  "bind_into(...) callable must take yorch::direct_out<T> as its last parameter"
  "bind_into wrong output parameter"
)

yorch_expect_compile_failure(
  bind_member_not_member
  "bind_member(...) requires a non-static member function pointer"
  "bind_member non-member callable"
)

yorch_expect_compile_failure(
  bind_member_direct_output
  "bind_member(...) does not accept direct-output member functions; use bind_into_member(...) instead"
  "bind_member direct-output member"
)

yorch_expect_compile_failure(
  bind_member_arity_mismatch
  "bind_member(...) requires one receiver binding plus exactly one spec per member-function parameter"
  "bind_member arity mismatch"
)

yorch_expect_compile_failure(
  bind_into_member_void_output
  "bind_into_member<T>(...) requires a non-void owned payload type T"
  "bind_into_member void output"
)

yorch_expect_compile_failure(
  bind_into_member_reference_output
  "bind_into_member<T>(...) requires a non-void owned payload type T"
  "bind_into_member reference output"
)

yorch_expect_compile_failure(
  bind_into_member_not_member
  "bind_into_member(...) requires a non-static member function pointer"
  "bind_into_member non-member callable"
)

yorch_expect_compile_failure(
  bind_into_member_missing_output_parameter
  "bind_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>"
  "bind_into_member missing output parameter"
)

yorch_expect_compile_failure(
  bind_into_member_arity_mismatch
  "bind_into_member(...) requires one receiver binding plus exactly one spec per non-output member-function parameter"
  "bind_into_member arity mismatch"
)

yorch_expect_compile_failure(
  bind_into_member_wrong_output_parameter
  "bind_into_member(...) callable must take yorch::direct_out<T> as its last parameter"
  "bind_into_member wrong output parameter"
)

yorch_expect_compile_failure(
  bind_forward_prev_void_output
  "bind_forward_prev<T>(...) requires a non-void owned payload type T"
  "bind_forward_prev void output"
)

yorch_expect_compile_failure(
  bind_forward_prev_reference_output
  "bind_forward_prev<T>(...) requires a non-void owned payload type T"
  "bind_forward_prev reference output"
)

yorch_expect_compile_failure(
  bind_forward_prev_direct_out_callable
  "bind_forward_prev<T>(...) does not accept callables with yorch::direct_out<T>; use bind_into(...) for direct-output materialization"
  "bind_forward_prev direct_out callable"
)

yorch_expect_compile_failure(
  bind_forward_prev_member_callable
  "bind_forward_prev<T>(...) does not support member function pointers in v1"
  "bind_forward_prev member callable"
)

yorch_expect_compile_failure(
  bind_forward_prev_prev_access_count
  "bind_forward_prev<T>(...) requires exactly one prev-access binding"
  "bind_forward_prev prev-access count"
)

yorch_expect_compile_failure(
  bind_forward_prev_borrow_prev
  "bind_forward_prev<T>(...) only supports borrow_prev_mut<T>() -> T& or consume_prev<T>() -> T&&"
  "bind_forward_prev borrow_prev binding"
)

yorch_expect_compile_failure(
  bind_forward_prev_copy_prev
  "bind_forward_prev<T>(...) only supports borrow_prev_mut<T>() -> T& or consume_prev<T>() -> T&&"
  "bind_forward_prev copy_prev binding"
)

yorch_expect_compile_failure(
  bind_forward_prev_consume_by_value
  "bind_forward_prev<T>(...) does not support consume_prev<T>() bound to T; use T&& if you want to forward the direct parent object identity"
  "bind_forward_prev consume by value"
)

yorch_expect_compile_failure(
  task_result_void
  "yorch::task_result<void> is not supported"
  "yorch::task_result<void>"
)

yorch_expect_compile_failure(
  compile_plan_forward_prev_root
  "compile_plan(...) does not allow forward-prev nodes at the root"
  "compile_plan forward_prev root"
)

yorch_expect_compile_failure(
  compile_plan_forward_prev_void_parent
  "compile_plan(...) requires forward-prev nodes to have a direct parent carrying a non-void payload"
  "compile_plan forward_prev void parent"
)

yorch_expect_compile_failure(
  forward_prev_parent_type_mismatch
  "compile_plan(...) requires forward-prev nodes to match the direct parent logical output type exactly"
  "forward_prev parent-type mismatch"
)

yorch_expect_compile_failure(
  compile_plan_retry_consume_prev
  "compile_plan(...) rejected the task tree because a task uses an invalid prev-access combination"
  "compile_plan retry consume_prev"
)

yorch_expect_compile_failure(
  compile_plan_fanout_invalid
  "compile_plan(...) rejected the task tree because the selected fanout policy is incompatible with child prev-access modes"
  "compile_plan fanout invalid"
)
