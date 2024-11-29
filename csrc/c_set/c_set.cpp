#include <pybind11/pybind11.h>
#include <cmath>
#include <set>
#include <cstdio>
namespace py = pybind11;
//T1->weight=frequency*delta_T, T2->last_accessed, T3->hash
// base = 0.9 
//g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` c_set.cpp -o c_set`python3-config --extension-suffix`



#define T1 double
#define T2 float
#define T3 long long
namespace constant {
    T1 base;
    T1 epsilon = 1e-9;
}
struct Item {
    T1 weight;
    T2 last_accessed;
    T3 hash;
    Item(T1 w, T2 l, T3 h) : weight(w), last_accessed(l), hash(h) {}
    bool operator <(const Item &rhs) const {
        auto p = weight, q = rhs.weight;
        if (last_accessed < rhs.last_accessed) q *= pow(constant::base, rhs.last_accessed - last_accessed);
        else p *= pow(constant::base, last_accessed - rhs.last_accessed);
        return p<q-constant::epsilon || (p<q+constant::epsilon && hash < rhs.hash);
    }
};
class c_set {

    std::set<Item> s;
public:
    c_set(T1 base) { constant::base = base; }
    void insert(T1 weight, T2 last_accessed, T3 hash) {
        s.emplace(weight, last_accessed, hash);
    }
    void erase(T1 weight, T2 last_accessed, T3 hash) {
        auto it = s.find({weight, last_accessed, hash});
        if (it != s.end()) s.erase(it);
    }
    T3 pop_first() {
        if (!s.empty()) {
            T3 hash = s.begin()->hash;
            s.erase(s.begin());
            return hash;
        }
        assert(0);
    }
};

PYBIND11_MODULE(c_set, m) {
    py::class_<c_set>(m, "c_set")
        .def(py::init<T1>())
        .def("insert", &c_set::insert)
        .def("erase", &c_set::erase)
        .def("pop_first", &c_set::pop_first);
}