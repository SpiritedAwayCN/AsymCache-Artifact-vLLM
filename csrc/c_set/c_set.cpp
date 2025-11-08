#include <pybind11/pybind11.h>
#include <cmath>
#include <set>
#include <cstdio>
#include <string>
#include <sstream>
#include <stdexcept>
namespace py = pybind11;
//T1->weight=frequency*delta_T, T1->last_accessed, T3->hash
// base = 0.9 
//g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` c_set.cpp -o c_set`python3-config --extension-suffix`



#define T1 double
// #define T1 float
#define T3 long long
namespace constant {
    T1 base;
    T1 epsilon = 1e-9;
    T1 TANH_DELTA, TANH_THETA;
}
struct Item {
    T1 weight;
    T1 last_accessed;
    T3 hash;
    Item(T1 w, T1 l, T3 h) : weight(w), last_accessed(l), hash(h) {}
    bool operator <(const Item &rhs) const {
        auto p = weight, q = rhs.weight;
        if (p < 1e9 && q < 1e9) {
            if (last_accessed < rhs.last_accessed) p *= pow(constant::base, rhs.last_accessed - last_accessed);
            else q *= pow(constant::base, last_accessed - rhs.last_accessed);
        }
        return p<q-constant::epsilon || (p<q+constant::epsilon && hash < rhs.hash);
    }
};

struct Item2{
    T1 weight;
    T1 last_accessed;
    T3 hash;
    T3 num_hashed_token;
    Item2(T1 w, T1 l, T3 h, T3 n) : weight(w), last_accessed(l), hash(h), num_hashed_token(n) {}
    bool operator <(const Item2 &rhs) const {
        auto p = weight, q = rhs.weight;
        if (last_accessed < rhs.last_accessed) p *= pow(constant::base, rhs.last_accessed - last_accessed);
        else q *= pow(constant::base, last_accessed - rhs.last_accessed);
        if (p-q < -constant::epsilon || p-q > constant::epsilon) return p<q;
        if (num_hashed_token != rhs.num_hashed_token) return num_hashed_token > rhs.num_hashed_token;
        return hash < rhs.hash;
    }
};

class c_set {
    std::set<Item> s;
public:
    c_set(T1 base) { constant::base = base; }
    void insert(T1 weight, T1 last_accessed, T3 hash) {
        s.emplace(weight, last_accessed, hash);
    }
    void erase(T1 weight, T1 last_accessed, T3 hash) {
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

class c_set2 {
    std::set<Item2> s;
public:
    c_set2(T1 base) { constant::base = base; }
    void insert(T1 weight, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        s.emplace(weight, last_accessed, hash, num_hashed_token);
    }
    void erase(T1 weight, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        auto it = s.find({weight, last_accessed, hash, num_hashed_token});
        if (it != s.end()) 
            s.erase(it);
        else{
            std::ostringstream oss;
            oss << "Item not found: " << weight << " " << last_accessed << " " << hash << " " << num_hashed_token;
            throw std::runtime_error(oss.str());
        }
    }
    T3 pop_first() {
        if (!s.empty()) {
            T3 hash = s.begin()->hash;
            s.erase(s.begin());
            return hash;
        }
        assert(0);
    }

    T3 size() {
        return static_cast<T3>(s.size());
    }
};
struct ItemTanh {
    T1 w1, w2, func;
    T1 last_accessed;
    T3 hash;
    ItemTanh(T1 w1, T1 w2, T1 func, T1 l, T3 h) : w1(w1), w2(w2), func(func), last_accessed(l), hash(h) {}
    bool operator <(const ItemTanh &rhs) const {
        auto h1 = w1, t1 = w2, p = h1 + t1;
        auto h2 = rhs.w1, t2 = rhs.w2, q = h2 + t2;
        if (last_accessed < rhs.last_accessed) {
            h1 *= exp(-(rhs.last_accessed - last_accessed) / constant::TANH_THETA);
            t1 = 1 - 2 / ((2/(1-t1)-1) * exp(-2*(rhs.last_accessed - last_accessed)/constant::TANH_THETA) + 1);
            p = h1 + t1;
        } else {
            h2 *= exp(-(last_accessed - rhs.last_accessed) / constant::TANH_THETA);
            t2 = 1 - 2 / ((2/(1-t2)-1) * exp(-2*(last_accessed - rhs.last_accessed)/constant::TANH_THETA) + 1);
            q = h2 + t2;
        }
        p *= func;
        q *= rhs.func;
        return p<q-constant::epsilon || (p<q+constant::epsilon && hash < rhs.hash);
    }
};

struct Item2Tanh {
    T1 w1, w2;
    T1 last_accessed;
    T3 hash;
    T3 num_hashed_token;
    Item2Tanh(T1 w1, T1 w2, T1 l, T3 h, T3 n) : w1(w1), w2(w2), last_accessed(l), hash(h), num_hashed_token(n) {}
    bool operator <(const Item2Tanh &rhs) const {
        auto h1 = w1, t1 = w2, p = h1 + t1;
        auto h2 = rhs.w1, t2 = rhs.w2, q = h2 + t2;
        if (last_accessed < rhs.last_accessed) {
            h1 *= exp(-(rhs.last_accessed - last_accessed) / constant::TANH_THETA);
            t1 = 1 - 2 / ((2/(1-t1)-1) * exp(-2*(rhs.last_accessed - last_accessed)/constant::TANH_THETA) + 1);
            p = h1 + t1;
        } else {
            h2 *= exp(-(last_accessed - rhs.last_accessed) / constant::TANH_THETA);
            t2 = 1 - 2 / ((2/(1-t2)-1) * exp(-2*(last_accessed - rhs.last_accessed)/constant::TANH_THETA) + 1);
            q = h2 + t2;
        }
        if (p-q < -constant::epsilon || p-q > constant::epsilon) return p<q;
        if (num_hashed_token != rhs.num_hashed_token) return num_hashed_token > rhs.num_hashed_token;
        return hash < rhs.hash;
    }
};
class c_set_tanh {
    std::set<ItemTanh> s;
public:
    c_set_tanh(T1 delta, T1 theta) {
        constant::TANH_DELTA=delta;
        constant::TANH_THETA=theta;
    }
    void insert(T1 w1, T1 w2, T1 func, T1 last_accessed, T3 hash) {
        s.emplace(w1, w2, func, last_accessed, hash);
    }
    void erase(T1 w1, T1 w2, T1 func, T1 last_accessed, T3 hash) {
        auto it = s.find({w1, w2, func, last_accessed, hash});
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
class c_set2_tanh {
    std::set<Item2Tanh> s;
public:
    c_set2_tanh(T1 delta, T1 theta) {
        constant::TANH_DELTA=delta;
        constant::TANH_THETA=theta;
    }
    void insert(T1 w1, T1 w2, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        s.emplace(w1, w2, last_accessed, hash, num_hashed_token);
    }
    void erase(T1 w1, T1 w2, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        auto it = s.find({w1, w2, last_accessed, hash, num_hashed_token});
        if (it != s.end()) 
            s.erase(it);
        else{
            std::ostringstream oss;
            oss << "Item not found: " << w1 << " " << w2 << " " << last_accessed << " " << hash << " " << num_hashed_token;
            throw std::runtime_error(oss.str());
        }
    }
    T3 pop_first() {
        if (!s.empty()) {
            T3 hash = s.begin()->hash;
            s.erase(s.begin());
            return hash;
        }
        assert(0);
    }

    T3 size() {
        return static_cast<T3>(s.size());
    }
};

PYBIND11_MODULE(c_set, m) {
    py::class_<c_set>(m, "c_set")
        .def(py::init<T1>())
        .def("insert", &c_set::insert)
        .def("erase", &c_set::erase)
        .def("pop_first", &c_set::pop_first);
    
    py::class_<c_set2>(m, "c_set_lfu")
        .def(py::init<T1>())
        .def("insert", &c_set2::insert)
        .def("erase", &c_set2::erase)
        .def("pop_first", &c_set2::pop_first)
        .def("size", &c_set2::size);

    py::class_<c_set_tanh>(m, "c_set_tanh")
        .def(py::init<T1, T1>())
        .def("insert", &c_set_tanh::insert)
        .def("erase", &c_set_tanh::erase)
        .def("pop_first", &c_set_tanh::pop_first);
    
    py::class_<c_set2_tanh>(m, "c_set_lfu_tanh")
        .def(py::init<T1, T1>())
        .def("insert", &c_set2_tanh::insert)
        .def("erase", &c_set2_tanh::erase)
        .def("pop_first", &c_set2_tanh::pop_first)
        .def("size", &c_set2_tanh::size);
}