package org.example.authservice.repository;

import org.example.authservice.models.Users;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface UserRepository extends JpaRepository<Users, String> {
    @Query("SELECT u FROM Users u WHERE u.phone = :phone")
    @org.springframework.transaction.annotation.Transactional(readOnly = true)
    Users findByPhone(@Param("phone") String phone);

    @Query("SELECT u FROM Users u WHERE LOWER(u.email) = LOWER(:email)")
    @org.springframework.transaction.annotation.Transactional(readOnly = true)
    Users findByEmail(@Param("email") String email);

    @Query("SELECT COUNT(u) FROM Users u WHERE u.role = :role")
    long countByRole(@Param("role") org.example.authservice.models.Role role);
}
