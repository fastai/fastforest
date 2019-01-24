#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "fastforest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fastforest, m) {
    py::class_<Node>(m, "Node")
        .def_readonly("left", &Node::left)
        .def_readonly("right", &Node::right)
        .def_readonly("start", &Node::start)
        .def_readonly("nrows", &Node::nrows)
        .def_readonly("bestPred", &Node::bestPred)
        .def_readonly("cutoff", &Node::cutoff)
        .def_readonly("value", &Node::value)
        .def_readonly("gini", &Node::gini);

    py::class_<FastTree>(m, "FastTree")
        .def("predict", &FastTree::predict)
        .def_readonly("root", &FastTree::root);

    py::class_<FastForest>(m, "FastForest")
        .def(py::init<Mat,Vec>())
        .def("build", &FastForest::build)
        .def("predict", &FastForest::predict)
        .def("getTree", &FastForest::getTree); 

    m.def("trainFF", &trainFF);
}


