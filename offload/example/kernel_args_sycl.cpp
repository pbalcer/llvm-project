#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

int main() {
  sycl::queue q;

  int *usm_ptr = sycl::malloc_host<int>(1, q);
  *usm_ptr = -1;

  int host_data[1] = {5};
  sycl::buffer<int, 1> buf(host_data, sycl::range<1>(1));

  int val = 42;

  q.submit([&](sycl::handler &h) {
      auto buf_acc =
          buf.get_access<sycl::access::mode::read_write>(h);
      sycl::local_accessor<int, 1> local_mem(
          sycl::range<1>(64), h);
      sycl::local_accessor<int, 1> local_mem2(
          sycl::range<1>(64), h);

      h.parallel_for(sycl::nd_range<1>(1, 1),
                     [=](sycl::nd_item<1> item) {
          local_mem[0] = val;
          local_mem2[0] = val + 1;
          buf_acc[0] = buf_acc[0] + local_mem[0] + local_mem2[0];
          usm_ptr[0] = buf_acc[0];
      });
  });

  q.wait();

  int expected = 5 + 42 + 43;
  printf("usm_ptr[0] = %d (expected %d)\n", *usm_ptr, expected);

  sycl::free(usm_ptr, q);

  return 0;
}
