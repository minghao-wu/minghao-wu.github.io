---
layout: archive
title: "Blog Posts"
permalink: /posts/
author_profile: true
---

Besides writing research papers, I also enjoy sharing my thoughts and experiences through blog posts. These posts are not formal academic articles, but rather personal reflections on various topics related to my research and interests, so I will not rigorously acknowledge all the sources. I hope you find them interesting and thought-provoking!

{% include base_path %}

{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}

