
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <ygm/comm.hpp>

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  using point_type = int;

  auto my_distance = [](const point_type &a, const point_type &b) {
    return ((double)b - a);
  };

  saltatlas::dhnsw<uint32_t, point_type> my_dhnsw(my_distance, world, 3, 16);

  std::vector<uint32_t>   ids{0, 1, 2, 3};
  std::vector<point_type> points{10, 9, 8, 7};

  my_dhnsw.add_points(ids.begin(), ids.end(), points.begin(), points.end());

  return 0;
}
