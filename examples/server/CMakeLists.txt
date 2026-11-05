add_executable(parakeet-server
    main.cpp
    model_fetch.cpp
    openai_format.cpp)
target_link_libraries(parakeet-server PRIVATE parakeet)
target_include_directories(parakeet-server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party)
find_package(Threads REQUIRED)
target_link_libraries(parakeet-server PRIVATE Threads::Threads)
find_package(nlohmann_json REQUIRED)
target_link_libraries(parakeet-server PRIVATE nlohmann_json::nlohmann_json)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED IMPORTED_TARGET sentencepiece)
target_link_libraries(parakeet-server PRIVATE PkgConfig::SENTENCEPIECE)
