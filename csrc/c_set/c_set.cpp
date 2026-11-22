#include <pybind11/pybind11.h>
#include <cstdint>
#include <cmath>
#include <set>
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace py = pybind11;
//T1->weight=frequency*delta_T, T1->last_accessed, T3->hash
// base = 0.9 
//g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` c_set.cpp -o c_set`python3-config --extension-suffix`



#define T1 double
// #define T1 float
#define T3 long long
// namespace constant {
//     T1 base;
//     T1 epsilon = 1e-9;
//     // T1 TANH_DELTA, TANH_THETA;
// }
struct Item {
    T1 weight;
    T1 last_accessed;
    T3 hash;
    Item(T1 w, T1 l, T3 h) : weight(w), last_accessed(l), hash(h) {}
};
struct ItemComparator {
    T1 base;
    T1 epsilon;
    ItemComparator(T1 b, T1 e) : base(b), epsilon(e) {}
    bool operator ()(const Item &a, const Item &b) const {
        auto p = a.weight, q = b.weight;
        if (p < 1e9 && q < 1e9) {
            if (a.last_accessed < b.last_accessed) p *= pow(base, b.last_accessed - a.last_accessed);
            else q *= pow(base, a.last_accessed - b.last_accessed);
        }
        return p<q-epsilon || (p<q+epsilon && a.hash < b.hash);
    }
};

struct Item2 {
    T1 weight;
    T1 last_accessed;
    T3 hash;
    T3 num_hashed_token;
    Item2(T1 w, T1 l, T3 h, T3 n) : weight(w), last_accessed(l), hash(h), num_hashed_token(n) {}
};
struct Item2Comparator {
    T1 base;
    T1 epsilon;
    Item2Comparator(T1 b, T1 e) : base(b), epsilon(e) {}
    bool operator ()(const Item2 &a, const Item2 &b) const {
        auto p = a.weight, q = b.weight;
        if (a.last_accessed < b.last_accessed) p *= pow(base, b.last_accessed - a.last_accessed);
        else q *= pow(base, a.last_accessed - b.last_accessed);
        if (p-q < -epsilon || p-q > epsilon) return p<q;
        if (a.num_hashed_token != b.num_hashed_token) return a.num_hashed_token > b.num_hashed_token;
        return a.hash < b.hash;
    }
};

class c_list {
    std::vector<Item> v;
    std::vector<Item>::iterator min_ptr;
    ItemComparator cmp;
    T1 base;
    void _get_min() {
        min_ptr = v.end();
        for (auto p = v.begin(); p != v.end(); ++p)
            if (min_ptr == v.end() || cmp(*p, *min_ptr))
                min_ptr = p;
    }
public:
    c_list(T1 b): cmp(b, 1e-9), base(b) {}
    void insert(T1 weight, T1 last_accessed, T3 hash) {
        v.emplace_back(weight, last_accessed, hash);
    }
    void erase(T1 weight, T1 last_accessed, T3 hash) {
        Item it(weight, last_accessed, hash);
        for (auto p = v.begin(); p != v.end(); ++p)
            if (!cmp(*p, it) && !cmp(it, *p)) {
                v.erase(p);
                return;
            }
        std::ostringstream oss;
        oss << "Item not found in c_list : " << weight << " " << last_accessed << " " << hash;
        throw std::runtime_error(oss.str());
    }
    T3 pop_min() {
        _get_min();
        assert(min_ptr != v.end());
        T3 hash = min_ptr->hash;
        v.erase(min_ptr);
        return hash;
    }
    std::pair<T1, T1> min() {
        _get_min();
        assert(min_ptr != v.end());
        return std::make_pair(min_ptr->weight, min_ptr->last_accessed);
    }
    T3 size() {
        return static_cast<T3>(v.size());
    }
};

class c_list_with_num {
    std::vector<Item2> v;
    std::vector<Item2>::iterator min_ptr;
    Item2Comparator cmp;
    T1 base;
    void _get_min() {
        min_ptr = v.end();
        for (auto p = v.begin(); p != v.end(); ++p)
            if (min_ptr == v.end() || cmp(*p, *min_ptr))
                min_ptr = p;
    }
public:
    c_list_with_num(T1 b): cmp(b, 1e-9), base(b) {}
    void insert(T1 weight, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        v.emplace_back(weight, last_accessed, hash, num_hashed_token);
    }
    void erase(T1 weight, T1 last_accessed, T3 hash, T3 num_hashed_token) {
        Item2 it(weight, last_accessed, hash, num_hashed_token);
        for (auto p = v.begin(); p != v.end(); ++p)
            if (!cmp(*p, it) && !cmp(it, *p)) {
                v.erase(p);
                return;
            }
        std::ostringstream oss;
        oss << "Item not found in c_list_with_num : " << weight << " " << last_accessed << " " << hash;
        throw std::runtime_error(oss.str());
    }
    T3 pop_min() {
        _get_min();
        assert(min_ptr != v.end());
        T3 hash = min_ptr->hash;
        v.erase(min_ptr);
        return hash;
    }
    std::pair<T1, T1> min() {
        _get_min();
        assert(min_ptr != v.end());
        return std::make_pair(min_ptr->weight, min_ptr->last_accessed);
    }
    T3 size() {
        return static_cast<T3>(v.size());
    }
};

class c_set {
    std::set<Item, ItemComparator> s;
    T1 base;
public:
    c_set(T1 b) : s(ItemComparator(b, 1e-9)), base(b) {}
    void insert(T1 weight, T1 last_accessed, T3 hash) {
        s.emplace(weight, last_accessed, hash);
    }
    void erase(T1 weight, T1 last_accessed, T3 hash) {
        auto it = s.find({weight, last_accessed, hash});
        if (it != s.end()) {
            s.erase(it);
        } else {
            std::ostringstream oss;
            oss << "Item not found: " << weight << " " << last_accessed << " " << hash;
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
    std::pair<T1, T1> first() {
        if (!s.empty()) {
            return std::make_pair(s.begin()->weight, s.begin()->last_accessed);
        }
        assert(0);
    }
    void display() {
        int w = 0;
        for (const auto &x : s) {
            w += 1;
            printf("<%.7lf,%.7lf,%lld> ", x.weight, x.last_accessed, x.hash);
            if (w % 5 == 0) puts("");
        }
        puts("");
    }

};

class c_set2 {
    std::set<Item2, Item2Comparator> s;
    T1 base;
public:
    c_set2(T1 b) : s(Item2Comparator(b, 1e-9)), base(b) {}
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
    std::pair<T1, T1> first() {
        if (!s.empty()) {
            return std::make_pair(s.begin()->weight, s.begin()->last_accessed);
        }
        assert(0);
    }

    T3 size() {
        return static_cast<T3>(s.size());
    }
};

struct HotPrefixBlock {
    T3 content_hash;
    T3 num_hashed_tokens;
    std::uint64_t inserted_epoch;
};

class HotPrefixEvictorCore {
    static constexpr T3 block_size = 64;
    static constexpr T3 initial_clock = 256;

    std::unordered_map<T3, HotPrefixBlock> blocks;
    std::unordered_map<T3, T3> hit_counts;
    std::uint64_t aging_epoch = 0;

    T3 clock(const HotPrefixBlock &block) const {
        const auto age = aging_epoch - block.inserted_epoch;
        return age >= static_cast<std::uint64_t>(initial_clock)
            ? 0
            : initial_clock - static_cast<T3>(age);
    }

public:
    void add(T3 block_id, T3 content_hash, T3 num_hashed_tokens) {
        auto result = hit_counts.emplace(content_hash, 0);
        if (!result.second) {
            ++result.first->second;
        }
        blocks.insert_or_assign(
            block_id,
            HotPrefixBlock{content_hash, num_hashed_tokens, aging_epoch});
    }

    void aging_all() {
        ++aging_epoch;
    }

    bool contains(T3 block_id) const {
        return blocks.find(block_id) != blocks.end();
    }

    std::pair<T3, T3> evict() {
        if (blocks.empty()) {
            throw std::runtime_error("No usable cache memory left");
        }

        auto chosen = blocks.end();
        T3 chosen_score = 0;
        T3 chosen_clock = 0;
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            const auto hit_count_it = hit_counts.find(it->second.content_hash);
            const T3 hit_count = hit_count_it == hit_counts.end()
                ? 0
                : hit_count_it->second;
            const T3 item_clock = clock(it->second);
            const T3 score = hit_count * block_size + item_clock;
            if (chosen == blocks.end()
                    || score < chosen_score
                    || (score == chosen_score
                        && item_clock < chosen_clock)
                    || (score == chosen_score
                        && item_clock == chosen_clock
                        && it->second.num_hashed_tokens
                            > chosen->second.num_hashed_tokens)
                    || (score == chosen_score
                        && item_clock == chosen_clock
                        && it->second.num_hashed_tokens
                            == chosen->second.num_hashed_tokens
                        && it->first < chosen->first)) {
                chosen = it;
                chosen_score = score;
                chosen_clock = item_clock;
            }
        }

        const T3 block_id = chosen->first;
        const T3 content_hash = chosen->second.content_hash;
        blocks.erase(chosen);
        hit_counts.erase(content_hash);
        return std::make_pair(block_id, content_hash);
    }

    T3 get_clock(T3 block_id) const {
        const auto it = blocks.find(block_id);
        if (it == blocks.end()) {
            throw std::runtime_error("Block not found in HotPrefixEvictorCore");
        }
        return clock(it->second);
    }

    T3 get_hit_count(T3 content_hash) const {
        const auto it = hit_counts.find(content_hash);
        return it == hit_counts.end() ? 0 : it->second;
    }

    void remove(T3 block_id) {
        if (blocks.erase(block_id) == 0) {
            throw std::runtime_error("Block not found in HotPrefixEvictorCore");
        }
    }

    T3 size() const {
        return static_cast<T3>(blocks.size());
    }
};
// struct ItemTanh {
//     T1 w1, w2, func;
//     T1 last_accessed;
//     T3 hash;
//     ItemTanh(T1 w1, T1 w2, T1 func, T1 l, T3 h) : w1(w1), w2(w2), func(func), last_accessed(l), hash(h) {}
//     bool operator <(const ItemTanh &rhs) const {
//         auto h1 = w1, t1 = w2, p = h1 + t1;
//         auto h2 = rhs.w1, t2 = rhs.w2, q = h2 + t2;
//         if (last_accessed != rhs.last_accessed) {
//             h1 *= exp(-(rhs.last_accessed - last_accessed) / constant::TANH_THETA);
//             t1 = 1 - 1 / ((1/(1-t1)-1) * exp(-2*(rhs.last_accessed - last_accessed)/constant::TANH_THETA) + 1);
//             p = h1 + t1;
//         }

//         // if (last_accessed < rhs.last_accessed) {
            
//         // } else {
//         //     h2 *= exp(-(last_accessed - rhs.last_accessed) / constant::TANH_THETA);
//         //     t2 = 1 - 1 / ((1/(1-t2)-1) * exp(-2*(last_accessed - rhs.last_accessed)/constant::TANH_THETA) + 1);
//         //     q = h2 + t2;
//         // }
//         p *= func;
//         q *= rhs.func;
//         return p<q-constant::epsilon || (p<q+constant::epsilon && hash < rhs.hash);
//     }
// };

// struct Item2Tanh {
//     T1 w1, w2;
//     T1 last_accessed;
//     T3 hash;
//     T3 num_hashed_token;
//     Item2Tanh(T1 w1, T1 w2, T1 l, T3 h, T3 n) : w1(w1), w2(w2), last_accessed(l), hash(h), num_hashed_token(n) {}
//     bool operator <(const Item2Tanh &rhs) const {
//         auto h1 = w1, t1 = w2, p = h1 + t1;
//         auto h2 = rhs.w1, t2 = rhs.w2, q = h2 + t2;
//         if (last_accessed < rhs.last_accessed) {
//             h1 *= exp(-(rhs.last_accessed - last_accessed) / constant::TANH_THETA);
//             t1 = 1 - 1 / ((1/(1-t1)-1) * exp(-2*(rhs.last_accessed - last_accessed)/constant::TANH_THETA) + 1);
//             p = h1 + t1;
//         } else {
//             h2 *= exp(-(last_accessed - rhs.last_accessed) / constant::TANH_THETA);
//             t2 = 1 - 1 / ((1/(1-t2)-1) * exp(-2*(last_accessed - rhs.last_accessed)/constant::TANH_THETA) + 1);
//             q = h2 + t2;
//         }
//         if (p-q < -constant::epsilon || p-q > constant::epsilon) return p<q;
//         if (num_hashed_token != rhs.num_hashed_token) return num_hashed_token > rhs.num_hashed_token;
//         return hash < rhs.hash;
//     }
// };
// class c_set_tanh {
//     std::set<ItemTanh> s;
// public:
//     c_set_tanh(T1 delta, T1 theta) {
//         constant::TANH_DELTA=delta;
//         constant::TANH_THETA=theta;
//     }
//     void insert(T1 w1, T1 w2, T1 func, T1 last_accessed, T3 hash) {
//         s.emplace(w1, w2, func, last_accessed, hash);
//     }
//     void erase(T1 w1, T1 w2, T1 func, T1 last_accessed, T3 hash) {
//         auto it = s.find({w1, w2, func, last_accessed, hash});
//         if (it != s.end()) 
//             s.erase(it);
//         else{
//             std::ostringstream oss;
//             oss << "Item not found: " << w1 << " " << w2 << " " << func << " " << last_accessed << " " << hash;
//             // print all entries in this container
//             // for (const auto& item : s) {
//             //     std::cout << "Existing item: " << item.w1 << " " << item.w2 << " " << item.func << " " << item.last_accessed << " " << item.hash << std::endl;
//             // }
//             throw std::runtime_error(oss.str());
//         }
//     }
//     T3 pop_first() {
//         if (!s.empty()) {
//             T3 hash = s.begin()->hash;
//             s.erase(s.begin());
//             return hash;
//         }
//         assert(0);
//     }
// };
// class c_set2_tanh {
//     std::set<Item2Tanh> s;
// public:
//     c_set2_tanh(T1 delta, T1 theta) {
//         constant::TANH_DELTA=delta;
//         constant::TANH_THETA=theta;
//     }
//     void insert(T1 w1, T1 w2, T1 last_accessed, T3 hash, T3 num_hashed_token) {
//         s.emplace(w1, w2, last_accessed, hash, num_hashed_token);
//     }
//     void erase(T1 w1, T1 w2, T1 last_accessed, T3 hash, T3 num_hashed_token) {
//         auto it = s.find({w1, w2, last_accessed, hash, num_hashed_token});
//         if (it != s.end()) 
//             s.erase(it);
//         else{
//             std::ostringstream oss;
//             oss << "Item not found: " << w1 << " " << w2 << " " << last_accessed << " " << hash << " " << num_hashed_token;
//             throw std::runtime_error(oss.str());
//         }
//     }
//     T3 pop_first() {
//         if (!s.empty()) {
//             T3 hash = s.begin()->hash;
//             s.erase(s.begin());
//             return hash;
//         }
//         assert(0);
//     }

//     T3 size() {
//         return static_cast<T3>(s.size());
//     }
// };

PYBIND11_MODULE(c_set, m) {
    py::class_<c_set>(m, "c_set")
        .def(py::init<T1>())
        .def("insert", &c_set::insert)
        .def("erase", &c_set::erase)
        .def("first", &c_set::first)
        .def("pop_first", &c_set::pop_first)
        .def("display", &c_set::display);
    
    py::class_<c_set2>(m, "c_set_lfu")
        .def(py::init<T1>())
        .def("insert", &c_set2::insert)
        .def("erase", &c_set2::erase)
        .def("first", &c_set2::first)
        .def("pop_first", &c_set2::pop_first)
        .def("size", &c_set2::size);
    
    py::class_<c_list>(m, "c_list")
        .def(py::init<T1>())
        .def("insert", &c_list::insert)
        .def("erase", &c_list::erase)
        .def("min", &c_list::min)
        .def("pop_min", &c_list::pop_min)
        .def("size", &c_list::size);
    
    py::class_<c_list_with_num>(m, "c_list_with_num")
        .def(py::init<T1>())
        .def("insert", &c_list_with_num::insert)
        .def("erase", &c_list_with_num::erase)
        .def("min", &c_list_with_num::min)
        .def("pop_min", &c_list_with_num::pop_min)
        .def("size", &c_list_with_num::size);

    py::class_<HotPrefixEvictorCore>(m, "HotPrefixEvictorCore")
        .def(py::init<>())
        .def("add", &HotPrefixEvictorCore::add)
        .def("aging_all", &HotPrefixEvictorCore::aging_all)
        .def("contains", &HotPrefixEvictorCore::contains)
        .def("evict", &HotPrefixEvictorCore::evict)
        .def("get_clock", &HotPrefixEvictorCore::get_clock)
        .def("get_hit_count", &HotPrefixEvictorCore::get_hit_count)
        .def("remove", &HotPrefixEvictorCore::remove)
        .def("size", &HotPrefixEvictorCore::size);


    // py::class_<c_set_tanh>(m, "c_set_tanh")
    //     .def(py::init<T1, T1>())
    //     .def("insert", &c_set_tanh::insert)
    //     .def("erase", &c_set_tanh::erase)
    //     .def("pop_first", &c_set_tanh::pop_first);
    
    // py::class_<c_set2_tanh>(m, "c_set_lfu_tanh")
    //     .def(py::init<T1, T1>())
    //     .def("insert", &c_set2_tanh::insert)
    //     .def("erase", &c_set2_tanh::erase)
    //     .def("pop_first", &c_set2_tanh::pop_first)
    //     .def("size", &c_set2_tanh::size);
}