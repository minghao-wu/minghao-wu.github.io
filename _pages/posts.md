---
layout: archive
title: "Blog Posts"
permalink: /posts/
author_profile: true
---

Besides writing research papers, I also enjoy sharing my thoughts and experiences through blog posts.
 
{% include base_path %}

{% for post in site.posts reversed %}
  {% include archive-single.html %}
{% endfor %}

