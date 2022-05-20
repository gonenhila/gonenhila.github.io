---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---


Full list of publications can be found in <u><a href="https://scholar.google.com/citations?user=URThmtMAAAAJ&hl=en">my Google Scholar profile</a>.</u>


{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
