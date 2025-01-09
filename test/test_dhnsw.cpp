
#include <cstdlib>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/dnnd/dnnd_simple.hpp>

#include <hnswlib/space_l2.h>

using id_type    = int;
using point_type = int;
using dist_type  = int;

dist_type my_distance(const point_type &a, const point_type &b) {
  return std::abs(b - a);
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  auto my_distance_lambda = [](const int &a, const int &b) {
    return std::abs(b - a);
  };

  // saltatlas::dhnsw<int, int, int> my_dhnsw(my_distance_lambda, world, 4,
  // world.size());
  // saltatlas::dhnsw<int, int, int> my_dhnsw(my_distance_lambda, world,
  // saltatlas::dhnsw_params{8});
  saltatlas::dhnsw<id_type, point_type, dist_type> my_dhnsw(
      my_distance, world, saltatlas::dhnsw_params{8});

  std::vector<id_type>    ids;
  std::vector<point_type> points;

  for (int i = 0; i < 100 * world.rank() + 4; ++i) {
    ids.push_back(i * world.size() + world.rank());
    points.push_back(i * world.size() + world.rank());
  }

  my_dhnsw.add_points(ids.begin(), ids.end(), points.begin(), points.end());

  my_dhnsw.build();

  // Querying
  std::vector<point_type> queries({{1, 3, 4, 8}});
  const auto query_results = my_dhnsw.query(queries.begin(), queries.end(), 2);

  return 0;
}
